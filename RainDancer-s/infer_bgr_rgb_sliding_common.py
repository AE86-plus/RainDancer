#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch
import torch .nn as nn
from tqdm import tqdm

REPO_ROOT =Path (__file__ ).resolve ().parent
sys .path .insert (0 ,str (REPO_ROOT ))

from networks .model import RMFD
from trainer .tools import rgb2ycbcr ,calculate_psnr ,calculate_ssim
from trainer .utils_rmfd import count_network_parameters


def load_config (config_path :Path ):
    try :
        import commentjson as cjson
        with open (config_path ,'r',encoding ='utf-8')as f :
            return cjson .load (f )
    except Exception :
        with open (config_path ,'r',encoding ='utf-8')as f :
            return json .load (f )


def choose_checkpoint (path_or_dir ):
    path =Path (path_or_dir )
    if path .is_file ():
        return path
    best =path /'best.pth.tar'
    if best .exists ():
        return best
    candidates =sorted (path .glob ('*.pth*'))
    if candidates :
        return candidates [-1 ]
    raise FileNotFoundError (f'No checkpoint found at {path }')


def strip_ddp_prefix (state_dict ):
    if not isinstance (state_dict ,dict ):
        return state_dict
    new_sd ={}
    for key ,value in state_dict .items ():
        new_key =key
        for prefix in ('module.','_orig_mod.'):
            if new_key .startswith (prefix ):
                new_key =new_key [len (prefix ):]
        new_sd [new_key ]=value
    return new_sd


def natural_key (value ):
    return [int (part )if part .isdigit ()else part .lower ()for part in re .split (r'(\d+)',str (value ))]


def format_count (num ):
    num =float (num )
    if num >=1e12 :
        return f'{num /1e12 :.4f} T'
    if num >=1e9 :
        return f'{num /1e9 :.4f} G'
    if num >=1e6 :
        return f'{num /1e6 :.4f} M'
    if num >=1e3 :
        return f'{num /1e3 :.4f} K'
    return str (int (num ))


def safe_mean (values ):
    if not values :
        return None
    return round (float (np .mean (values )),6 )


def h5_rgb_to_bgr_uint8 (dataset ):
    arr =dataset [:].copy ()
    if arr .ndim !=3 or arr .shape [2 ]!=3 :
        raise ValueError (f'Expected HxWx3 image, got {arr .shape }')
    return np .ascontiguousarray (arr [:,:,::-1 ])


def bgr_uint8_to_rgb_uint8 (img_bgr ):
    return np .ascontiguousarray (img_bgr [:,:,::-1 ])


def bgr_uint8_to_tensor01 (img_bgr ):
    return torch .from_numpy (np .ascontiguousarray (img_bgr )).permute (2 ,0 ,1 ).float ()/255.0


def bgr_tensor_to_rgb_uint8 (tensor ):
    arr =tensor .detach ().cpu ().clamp (0 ,1 ).permute (1 ,2 ,0 ).numpy ()
    arr =(arr *255.0 ).round ().astype (np .uint8 )
    return np .ascontiguousarray (arr [:,:,::-1 ])


def save_rgb (path :Path ,image_rgb :np .ndarray ):
    path .parent .mkdir (parents =True ,exist_ok =True )
    cv2 .imwrite (str (path ),cv2 .cvtColor (image_rgb ,cv2 .COLOR_RGB2BGR ))


def compute_y_metrics (pred_rgb ,gt_rgb ):
    pred_y =rgb2ycbcr (pred_rgb ,only_y =True )
    gt_y =rgb2ycbcr (gt_rgb ,only_y =True )
    return float (calculate_psnr (gt_y ,pred_y ,border =0 )),float (calculate_ssim (gt_y ,pred_y ,border =0 ))


def get_starts (length ,crop_size ,stride ):
    if length <=crop_size :
        return [0 ]
    starts =list (range (0 ,length -crop_size +1 ,stride ))
    if starts [-1 ]!=length -crop_size :
        starts .append (length -crop_size )
    return starts


class FlopCounter :
    def __init__ (self ):
        self .flops =0
        self .handles =[]

    def close (self ):
        for handle in self .handles :
            handle .remove ()
        self .handles =[]

    def add_hooks_for_module (self ,module ):
        for submodule in module .modules ():
            if isinstance (submodule ,nn .Conv2d ):
                self .handles .append (submodule .register_forward_hook (self .conv2d_hook ))
            elif isinstance (submodule ,nn .ConvTranspose2d ):
                self .handles .append (submodule .register_forward_hook (self .conv_transpose2d_hook ))
            elif isinstance (submodule ,nn .Linear ):
                self .handles .append (submodule .register_forward_hook (self .linear_hook ))
            elif isinstance (submodule ,(nn .BatchNorm2d ,nn .InstanceNorm2d ,nn .GroupNorm )):
                self .handles .append (submodule .register_forward_hook (self .norm_hook ))
            elif isinstance (submodule ,(nn .ReLU ,nn .ReLU6 ,nn .PReLU ,nn .LeakyReLU ,nn .Sigmoid ,nn .Tanh )):
                self .handles .append (submodule .register_forward_hook (self .activation_hook ))
            elif isinstance (submodule ,(nn .AvgPool2d ,nn .MaxPool2d ,nn .AdaptiveAvgPool2d )):
                self .handles .append (submodule .register_forward_hook (self .pool_hook ))
            elif isinstance (submodule ,nn .Upsample ):
                self .handles .append (submodule .register_forward_hook (self .upsample_hook ))

    def conv2d_hook (self ,module ,inputs ,output ):
        out =output
        batch ,out_c ,out_h ,out_w =out .shape
        kernel_mul =module .kernel_size [0 ]*module .kernel_size [1 ]*(module .in_channels //module .groups )
        bias_ops =1 if module .bias is not None else 0
        macs =batch *out_c *out_h *out_w *(kernel_mul +bias_ops )
        self .flops +=2 *macs

    def conv_transpose2d_hook (self ,module ,inputs ,output ):
        out =output
        batch ,out_c ,out_h ,out_w =out .shape
        kernel_mul =module .kernel_size [0 ]*module .kernel_size [1 ]*(module .in_channels //module .groups )
        bias_ops =1 if module .bias is not None else 0
        macs =batch *out_c *out_h *out_w *(kernel_mul +bias_ops )
        self .flops +=2 *macs

    def linear_hook (self ,module ,inputs ,output ):
        x =inputs [0 ]
        batch =x .shape [0 ]if x .ndim >1 else 1
        macs =batch *module .in_features *module .out_features
        self .flops +=2 *macs
        if module .bias is not None :
            self .flops +=batch *module .out_features

    def norm_hook (self ,module ,inputs ,output ):
        self .flops +=output .numel ()*2

    def activation_hook (self ,module ,inputs ,output ):
        self .flops +=output .numel ()

    def pool_hook (self ,module ,inputs ,output ):
        if isinstance (module ,(nn .AvgPool2d ,nn .MaxPool2d )):
            kernel_size =module .kernel_size
            if isinstance (kernel_size ,tuple ):
                kernel_ops =kernel_size [0 ]*kernel_size [1 ]
            else :
                kernel_ops =kernel_size *kernel_size
            self .flops +=output .numel ()*kernel_ops
        else :
            self .flops +=output .numel ()

    def upsample_hook (self ,module ,inputs ,output ):
        mode =getattr (module ,'mode','nearest')
        scale =1 if mode =='nearest'else 8
        self .flops +=output .numel ()*scale


def build_model (config ,checkpoint_path ,gpu ):
    if not torch .cuda .is_available ():
        raise RuntimeError ('CUDA is required for this script.')
    torch .cuda .set_device (gpu )
    RMFD .load_ddp =lambda self :None # type: ignore
    opt =SimpleNamespace (lr =1e-4 ,beta1 =0.5 ,beta2 =0.999 )
    model =RMFD (config ,opt ,gpu )
    checkpoint =torch .load (str (checkpoint_path ),map_location ='cpu')
    for name in model .train_model_names :
        if isinstance (name ,str )and name in checkpoint :
            getattr (model ,name ).load_state_dict (strip_ddp_prefix (checkpoint [name ]),strict =False )
    for name in model .eval_model_names :
        if isinstance (name ,str ):
            getattr (model ,name ).eval ()
    return model


def count_total_parameters (model ):
    seen =set ()
    total =0
    for name in model .train_model_names :
        if isinstance (name ,str ):
            module =getattr (model ,name )
            if id (module )in seen :
                continue
            seen .add (id (module ))
            total +=count_network_parameters (module )
    return int (total )


def compute_model_flops (model ,rainy ,event ,clean ):
    counter =FlopCounter ()
    seen =set ()
    for name in model .train_model_names :
        if isinstance (name ,str ):
            module =getattr (model ,name )
            if id (module )in seen :
                continue
            seen .add (id (module ))
            counter .add_hooks_for_module (module )
    with torch .no_grad ():
        model .set_input_test ({'Rain_frame':rainy ,'Rain_event':event ,'clean':clean })
        model .forward ()
    flops =int (counter .flops )
    counter .close ()
    return flops


def compute_model_flops_at_size (model ,input_size ,gpu ):
    device =torch .device (f'cuda:{gpu }')
    rainy =torch .zeros ((1 ,9 ,input_size ,input_size ),device =device ,dtype =torch .float32 )
    event =torch .zeros ((1 ,20 ,input_size ,input_size ),device =device ,dtype =torch .float32 )
    clean =torch .zeros ((1 ,3 ,input_size ,input_size ),device =device ,dtype =torch .float32 )
    return compute_model_flops (model ,rainy ,event ,clean )


@torch .no_grad ()
def sliding_infer_both (model ,rainy_9 ,event_20 ,clean_3 ,gpu ,crop_size ,stride ,patch_bs ,use_amp ):
    if rainy_9 .dim ()==3 :
        rainy_9 =rainy_9 .unsqueeze (0 )
    if event_20 .dim ()==3 :
        event_20 =event_20 .unsqueeze (0 )
    if clean_3 .dim ()==3 :
        clean_3 =clean_3 .unsqueeze (0 )

    device =torch .device (f'cuda:{gpu }')
    rainy_9 =rainy_9 .to (device ,non_blocking =True )
    event_20 =event_20 .to (device ,non_blocking =True )
    clean_3 =clean_3 .to (device ,non_blocking =True )

    _ ,_ ,height ,width =rainy_9 .shape
    ys =get_starts (height ,crop_size ,stride )
    xs =get_starts (width ,crop_size ,stride )

    acc_bg =torch .zeros ((1 ,3 ,height ,width ),device =device ,dtype =torch .float32 )
    acc_rain =torch .zeros ((1 ,3 ,height ,width ),device =device ,dtype =torch .float32 )
    weight =torch .zeros ((1 ,1 ,height ,width ),device =device ,dtype =torch .float32 )

    coords =[(y0 ,x0 )for y0 in ys for x0 in xs ]
    for start in range (0 ,len (coords ),patch_bs ):
        batch_coords =coords [start :start +patch_bs ]
        rainy_list =[]
        event_list =[]
        clean_list =[]
        for y0 ,x0 in batch_coords :
            y1 =y0 +crop_size
            x1 =x0 +crop_size
            rainy_list .append (rainy_9 [:,:,y0 :y1 ,x0 :x1 ])
            event_list .append (event_20 [:,:,y0 :y1 ,x0 :x1 ])
            clean_list .append (clean_3 [:,:,y0 :y1 ,x0 :x1 ])

        rainy_batch =torch .cat (rainy_list ,dim =0 )
        event_batch =torch .cat (event_list ,dim =0 )
        clean_batch =torch .cat (clean_list ,dim =0 )
        input_data ={'Rain_frame':rainy_batch ,'Rain_event':event_batch ,'clean':clean_batch }

        if use_amp :
            with torch .cuda .amp .autocast ():
                model .set_input_test (input_data )
                model .forward ()
        else :
            model .set_input_test (input_data )
            model .forward ()

        pred_bg_batch =model .Pred_bg .detach ().float ()
        pred_rain_batch =model .Pred_rl .detach ().float ()

        for idx ,(y0 ,x0 )in enumerate (batch_coords ):
            y1 =y0 +crop_size
            x1 =x0 +crop_size
            acc_bg [:,:,y0 :y1 ,x0 :x1 ]+=pred_bg_batch [idx :idx +1 ]
            acc_rain [:,:,y0 :y1 ,x0 :x1 ]+=pred_rain_batch [idx :idx +1 ]
            weight [:,:,y0 :y1 ,x0 :x1 ]+=1.0

    pred_bg =(acc_bg /(weight +1e-8 )).clamp (0 ,1 )
    pred_rain =(acc_rain /(weight +1e-8 )).clamp (0 ,1 )
    return pred_bg ,pred_rain ,len (coords )


def get_scene_names (h5f ,spec ,config ):
    dtype =spec ['type']
    if dtype =='rainsyn':
        scenes =sorted (h5f ['input'].keys (),key =natural_key )
    else :
        preferred =spec .get ('scene_types')or config .get ('scene_types')
        if preferred :
            scenes =[scene for scene in preferred if scene in h5f ]
        else :
            scenes =sorted (h5f .keys (),key =natural_key )
    return scenes


def get_dataset_views (h5f ,spec ,scene ):
    dtype =spec ['type']
    if dtype =='legacy_paired':
        rainy_group =h5f [scene ]['1']['rainy']
        gt_group =h5f [scene ]['gt']
        voxel_group =h5f [scene ]['1']['voxel']
        frame_keys =sorted (rainy_group .keys (),key =natural_key )
    elif dtype =='rainsyn':
        rainy_group =h5f ['input'][scene ]
        gt_group =h5f ['processed'][scene ]
        voxel_group =rainy_group ['voxel']
        frame_keys =sorted ([key for key in rainy_group .keys ()if key !='voxel'],key =natural_key )
    else :
        raise ValueError (f'Unknown dataset type: {dtype }')
    voxel_keys =sorted (voxel_group .keys (),key =natural_key )
    return rainy_group ,gt_group ,voxel_group ,frame_keys ,voxel_keys


def write_reports (output_dir ,summary ,sample_rows ,scene_rows ):
    output_dir .mkdir (parents =True ,exist_ok =True )
    (output_dir /'summary.json').write_text (json .dumps (summary ,indent =2 ,ensure_ascii =False ),encoding ='utf-8')
    with open (output_dir /'summary.txt','w',encoding ='utf-8')as f :
        for key ,value in summary .items ():
            if isinstance (value ,(dict ,list )):
                f .write (f'{key }: {json .dumps (value ,ensure_ascii =False )}\n')
            else :
                f .write (f'{key }: {value }\n')

    sample_fields =[
    'scene','sample',
    'input_psnr_y','input_ssim_y',
    'pred_bg_psnr_y','pred_bg_ssim_y',
    'pred_rain_psnr_y','pred_rain_ssim_y',
    'input_path','gt_path','pred_bg_path','pred_rain_path',
    ]
    with open (output_dir /'sample_metrics.csv','w',newline ='',encoding ='utf-8')as f :
        writer =csv .DictWriter (f ,fieldnames =sample_fields )
        writer .writeheader ()
        writer .writerows (sample_rows )

    scene_fields =[
    'scene','num_samples',
    'avg_input_psnr_y','avg_input_ssim_y',
    'avg_pred_bg_psnr_y','avg_pred_bg_ssim_y',
    'avg_pred_rain_psnr_y','avg_pred_rain_ssim_y',
    ]
    with open (output_dir /'scene_metrics.csv','w',newline ='',encoding ='utf-8')as f :
        writer =csv .DictWriter (f ,fieldnames =scene_fields )
        writer .writeheader ()
        writer .writerows (scene_rows )


def run_one_dataset (spec ,args ,output_root :Path ):
    config_path =Path (spec ['config'])
    checkpoint_path =choose_checkpoint (spec ['checkpoint'])
    config =load_config (config_path )

    print ('\n'+'='*100 )
    print (f"Dataset     : {spec ['name']}")
    print (f"Type        : {spec ['type']}")
    print (f"Config      : {config_path }")
    print (f"Checkpoint  : {checkpoint_path }")
    print (f"Test H5     : {spec ['h5']}")
    print (f"Stride      : {args .stride }")
    print ('='*100 )

    model =build_model (config ,checkpoint_path ,args .gpu )
    total_params =count_total_parameters (model )
    flops_value =compute_model_flops_at_size (model ,args .flops_size ,args .gpu )

    dataset_output =output_root /spec ['name']
    image_root =dataset_output /'images'
    image_root .mkdir (parents =True ,exist_ok =True )

    sample_rows =[]
    scene_rows =[]
    all_input_psnr =[]
    all_input_ssim =[]
    all_bg_psnr =[]
    all_bg_ssim =[]
    all_rain_psnr =[]
    all_rain_ssim =[]
    processed_scenes =[]
    example_patch_count =None

    with h5py .File (spec ['h5'],'r')as h5f :
        scenes =get_scene_names (h5f ,spec ,config )
        if args .max_scenes >0 :
            scenes =scenes [:args .max_scenes ]

        for scene in scenes :
            rainy_group ,gt_group ,voxel_group ,frame_keys ,voxel_keys =get_dataset_views (h5f ,spec ,scene )
            num_samples =min (len (frame_keys )-2 ,len (voxel_keys )-1 )
            if num_samples <=0 :
                print (f'[Skip] {scene }: not enough frames/events')
                continue
            center_indices =list (range (1 ,num_samples +1 ))
            if args .max_samples_per_scene >0 :
                center_indices =center_indices [:args .max_samples_per_scene ]
            if not center_indices :
                continue

            processed_scenes .append (scene )
            scene_input_psnr =[]
            scene_input_ssim =[]
            scene_bg_psnr =[]
            scene_bg_ssim =[]
            scene_rain_psnr =[]
            scene_rain_ssim =[]
            scene_dir =image_root /scene
            scene_dir .mkdir (parents =True ,exist_ok =True )

            for center_idx in tqdm (center_indices ,desc =f'{spec ["name"]}:{scene }',leave =False ):
                prev_key =frame_keys [center_idx -1 ]
                center_key =frame_keys [center_idx ]
                next_key =frame_keys [center_idx +1 ]

                rainy0_bgr =h5_rgb_to_bgr_uint8 (rainy_group [prev_key ])
                rainy1_bgr =h5_rgb_to_bgr_uint8 (rainy_group [center_key ])
                rainy2_bgr =h5_rgb_to_bgr_uint8 (rainy_group [next_key ])

                gt_bgr =None
                if gt_group is not None and center_key in gt_group :
                    gt_bgr =h5_rgb_to_bgr_uint8 (gt_group [center_key ])

                rainy_9 =torch .cat ([
                bgr_uint8_to_tensor01 (rainy0_bgr ),
                bgr_uint8_to_tensor01 (rainy1_bgr ),
                bgr_uint8_to_tensor01 (rainy2_bgr ),
                ],dim =0 )
                event_20 =torch .cat ([
                torch .from_numpy (voxel_group [voxel_keys [center_idx -1 ]][:].astype (np .float32 )),
                torch .from_numpy (voxel_group [voxel_keys [center_idx ]][:].astype (np .float32 )),
                ],dim =0 )
                clean_3 =bgr_uint8_to_tensor01 (gt_bgr if gt_bgr is not None else rainy1_bgr )

                pred_bg ,pred_rain ,patch_count =sliding_infer_both (
                model =model ,
                rainy_9 =rainy_9 ,
                event_20 =event_20 ,
                clean_3 =clean_3 ,
                gpu =args .gpu ,
                crop_size =args .crop_size ,
                stride =args .stride ,
                patch_bs =args .patch_batch ,
                use_amp =bool (args .amp ),
                )
                if example_patch_count is None :
                    example_patch_count =patch_count

                input_rgb =bgr_uint8_to_rgb_uint8 (rainy1_bgr )
                gt_rgb =bgr_uint8_to_rgb_uint8 (gt_bgr )if gt_bgr is not None else None
                pred_bg_rgb =bgr_tensor_to_rgb_uint8 (pred_bg [0 ])
                pred_rain_rgb =bgr_tensor_to_rgb_uint8 (pred_rain [0 ])

                input_path =Path ('images')/scene /f'{center_key }_input.png'
                gt_path =Path ('images')/scene /f'{center_key }_gt.png'
                pred_bg_path =Path ('images')/scene /f'{center_key }_pred_bg.png'
                pred_rain_path =Path ('images')/scene /f'{center_key }_pred_rain.png'

                if args .save_images :
                    save_rgb (dataset_output /input_path ,input_rgb )
                    save_rgb (dataset_output /pred_bg_path ,pred_bg_rgb )
                    save_rgb (dataset_output /pred_rain_path ,pred_rain_rgb )
                    if gt_rgb is not None :
                        save_rgb (dataset_output /gt_path ,gt_rgb )

                row ={
                'scene':scene ,
                'sample':center_key ,
                'input_psnr_y':'',
                'input_ssim_y':'',
                'pred_bg_psnr_y':'',
                'pred_bg_ssim_y':'',
                'pred_rain_psnr_y':'',
                'pred_rain_ssim_y':'',
                'input_path':str (input_path ),
                'gt_path':str (gt_path )if gt_rgb is not None else '',
                'pred_bg_path':str (pred_bg_path ),
                'pred_rain_path':str (pred_rain_path ),
                }

                if gt_rgb is not None :
                    gt_rain_rgb =np .clip (input_rgb .astype (np .int16 )-gt_rgb .astype (np .int16 ),0 ,255 ).astype (np .uint8 )
                    input_psnr ,input_ssim =compute_y_metrics (input_rgb ,gt_rgb )
                    bg_psnr ,bg_ssim =compute_y_metrics (pred_bg_rgb ,gt_rgb )
                    rain_psnr ,rain_ssim =compute_y_metrics (pred_rain_rgb ,gt_rain_rgb )

                    row .update ({
                    'input_psnr_y':f'{input_psnr :.6f}',
                    'input_ssim_y':f'{input_ssim :.6f}',
                    'pred_bg_psnr_y':f'{bg_psnr :.6f}',
                    'pred_bg_ssim_y':f'{bg_ssim :.6f}',
                    'pred_rain_psnr_y':f'{rain_psnr :.6f}',
                    'pred_rain_ssim_y':f'{rain_ssim :.6f}',
                    })

                    scene_input_psnr .append (input_psnr )
                    scene_input_ssim .append (input_ssim )
                    scene_bg_psnr .append (bg_psnr )
                    scene_bg_ssim .append (bg_ssim )
                    scene_rain_psnr .append (rain_psnr )
                    scene_rain_ssim .append (rain_ssim )
                    all_input_psnr .append (input_psnr )
                    all_input_ssim .append (input_ssim )
                    all_bg_psnr .append (bg_psnr )
                    all_bg_ssim .append (bg_ssim )
                    all_rain_psnr .append (rain_psnr )
                    all_rain_ssim .append (rain_ssim )

                sample_rows .append (row )

            scene_rows .append ({
            'scene':scene ,
            'num_samples':len (center_indices ),
            'avg_input_psnr_y':''if not scene_input_psnr else f'{np .mean (scene_input_psnr ):.6f}',
            'avg_input_ssim_y':''if not scene_input_ssim else f'{np .mean (scene_input_ssim ):.6f}',
            'avg_pred_bg_psnr_y':''if not scene_bg_psnr else f'{np .mean (scene_bg_psnr ):.6f}',
            'avg_pred_bg_ssim_y':''if not scene_bg_ssim else f'{np .mean (scene_bg_ssim ):.6f}',
            'avg_pred_rain_psnr_y':''if not scene_rain_psnr else f'{np .mean (scene_rain_psnr ):.6f}',
            'avg_pred_rain_ssim_y':''if not scene_rain_ssim else f'{np .mean (scene_rain_ssim ):.6f}',
            })

    summary ={
    'dataset':spec ['name'],
    'dataset_type':spec ['type'],
    'config':str (config_path ),
    'checkpoint':str (checkpoint_path ),
    'test_h5':str (spec ['h5']),
    'paired_gt':True ,
    'crop_size':int (args .crop_size ),
    'stride':int (args .stride ),
    'patch_batch':int (args .patch_batch ),
    'processed_scenes':processed_scenes ,
    'num_scenes':len (processed_scenes ),
    'num_samples':len (sample_rows ),
    'avg_input_psnr_y':safe_mean (all_input_psnr ),
    'avg_input_ssim_y':safe_mean (all_input_ssim ),
    'avg_pred_bg_psnr_y':safe_mean (all_bg_psnr ),
    'avg_pred_bg_ssim_y':safe_mean (all_bg_ssim ),
    'avg_pred_rain_psnr_y':safe_mean (all_rain_psnr ),
    'avg_pred_rain_ssim_y':safe_mean (all_rain_ssim ),
    'parameters':total_params ,
    'parameters_human':format_count (total_params ),
    'flops_per_128_window':int (flops_value ),
    'flops_per_128_window_human':format_count (flops_value ),
    'flops_window_size':int (args .flops_size ),
    'example_patch_count_per_image':example_patch_count ,
    'output_dir':str (dataset_output ),
    }
    write_reports (dataset_output ,summary ,sample_rows ,scene_rows )

    print (f"[Done] {spec ['name']}")
    if summary ['avg_pred_bg_psnr_y']is not None :
        print (f"  pred_bg PSNR(Y): {summary ['avg_pred_bg_psnr_y']}")
        print (f"  pred_bg SSIM(Y): {summary ['avg_pred_bg_ssim_y']}")
    print (f"  parameters     : {summary ['parameters_human']}")
    print (f"  flops/window   : {summary ['flops_per_128_window_human']}")
    print (f"  results        : {dataset_output }")

    del model
    torch .cuda .empty_cache ()


def build_parser (title ,default_output_root ,default_stride ):
    parser =argparse .ArgumentParser (description =title )
    parser .add_argument ('--gpu',type =int ,default =0 )
    parser .add_argument ('--crop-size',type =int ,default =128 )
    parser .add_argument ('--stride',type =int ,default =default_stride )
    parser .add_argument ('--patch-batch',type =int ,default =4 )
    parser .add_argument ('--amp',action ='store_true')
    parser .add_argument ('--flops-size',type =int ,default =128 )
    parser .add_argument ('--output-root',type =str ,default =str (default_output_root ))
    parser .add_argument ('--max-scenes',type =int ,default =0 )
    parser .add_argument ('--max-samples-per-scene',type =int ,default =0 )
    parser .add_argument ('--save-images',type =int ,default =1 )
    return parser


def run_collection (title ,dataset_specs ,default_output_root ,default_stride ):
    parser =build_parser (title ,default_output_root ,default_stride )
    args =parser .parse_args ()
    torch .backends .cudnn .benchmark =True
    output_root =Path (args .output_root )
    output_root .mkdir (parents =True ,exist_ok =True )

    print ('='*100 )
    print (title )
    print (f'Output Root : {output_root }')
    print (f'Crop Size   : {args .crop_size }')
    print (f'Stride      : {args .stride }')
    print (f'Patch Batch : {args .patch_batch }')
    print ('='*100 )

    for spec in dataset_specs :
        run_one_dataset (spec ,args ,output_root )

    print ('\nAll datasets finished.')
    print (f'Results root: {output_root }')
