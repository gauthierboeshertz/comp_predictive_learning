import math
import random
from PIL import Image, ImageDraw
import torch
import numpy as np
import copy 

NUMBER_OF_PRIMITIVES = 7
RETRY_LIMIT = 100
MIN_CLEARANCE = 7
SKETCH_WIDTH = 3
VISUALIZATION_COLOR_PALETTE = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'pink', 'brown', 'gray']

SEMI_CIRCLE_START = 170

def randint_from_union(a, b):
    assert 0 < a <= b, "Require 0 < a <= b"
    left_range = list(range(-b, -a + 1))   
    right_range = list(range(a, b + 1))    
    full_range = left_range + right_range
    return random.choice(full_range)


def draw_shapes_on_image(shapes, canvas_size, width=SKETCH_WIDTH, bg_color='black', shape_color='white',colorized=False):
    if shape_color  != 'white' or colorized:
        img = Image.new('RGB', (canvas_size, canvas_size), bg_color)
    else:
        img = Image.new('L', (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(img)
    for shape in shapes:
        if shape['type'] == 'line':
            draw.line([shape['start'], shape['end']], fill=shape_color, width=width)
        elif shape['type'] in ['circle', 'arc']:
            bb = [shape['center'][0] - shape['radius'], shape['center'][1] - shape['radius'],
                  shape['center'][0] + shape['radius'], shape['center'][1] + shape['radius']]
            if shape['type'] == 'circle':
                draw.ellipse(bb, outline=shape_color, width=width)
            else:
                draw.arc(bb, start=shape['start_angle'], end=shape['end_angle'], fill=shape_color, width=width)

        elif shape['type'] == 'regular_polygon':
            cx, cy = shape['center']
            r = shape['radius']
            n = shape['n_sides']
            rot = shape.get('rotation', 0)

            bb = [cx,cy,r]
            draw.regular_polygon(bb, n, rotation=rot, outline=shape_color, width=width)

    return img

def transform_shape(shape, offset, scale):
    new_shape = shape.copy()
    if shape['type'] == 'line':
        new_shape['start'] = (shape['start'][0] * scale + offset[0], shape['start'][1] * scale + offset[1])
        new_shape['end'] = (shape['end'][0] * scale + offset[0], shape['end'][1] * scale + offset[1])
    elif shape['type'] in ['circle', 'arc', 'regular_polygon']:
        new_shape['center'] = (shape['center'][0] * scale + offset[0], shape['center'][1] * scale + offset[1])
        new_shape['radius'] = shape['radius'] * scale

    return new_shape

      
def create_symmetrical_arc(radius, gap_angle):
    gap_center = 270
    half_gap = gap_angle / 2

    start_angle = gap_center + half_gap
    end_angle = gap_center - half_gap + 360 
    return {
        'type': 'arc',
        'center': (0, 0),
        'radius': radius,
        'start_angle': start_angle,
        'end_angle': end_angle,
    }


def make_sequence(sequence_length,
                  start_prim_index,
                  start_quadrant,
                  start_scale_idx,
                  start_color_idx,
                  primitive_names,
                  all_scales,
                  colors,
                  positions,
                  next_primitive_offset,
                  next_scale_offset,
                  next_position_offset,
                  next_color_offset,
                  noisy_in_quadrant_locations=False):
    
    drawing_metadata = []
    
    for i in range(sequence_length):
        prim_name = primitive_names[(start_prim_index + i * next_primitive_offset) % len(primitive_names)]
        scale = all_scales[(start_scale_idx + i * next_scale_offset) % len(all_scales)]
        color = colors[(start_color_idx + i * next_color_offset) % len(colors)]
        position = positions[(start_quadrant + i * next_position_offset) % len(positions)]
        if noisy_in_quadrant_locations:
            position = (position[0] + int(np.random.normal(0,2)), position[1] + int(np.random.normal(0,2)))
        drawing_metadata.append({'name': prim_name, 'position': position, 'scale': scale, 'color': color})

    return drawing_metadata

def generate_metadata(all_primitives,
                    all_scales,
                    colors,
                    positions,
                    canvas_size,
                    sequence_length=4,
                    next_primitive_offset=1,
                    next_scale_offset=1,
                    next_position_offset=1,
                    next_color_offset=1,
                    return_image=False,
                    noisy_in_quadrant_locations=False):
    
    drawing_metadata = []
    primitive_names = list(all_primitives.keys())
    possible_scales = list(all_scales)
    possible_primitive_indices = list(range(len(primitive_names)))
    possible_color_indices = list(range(len(colors)))


    possible_positions = positions
    
    start_prim_index = random.randint(0, len(possible_primitive_indices) - 1)
    start_quadrant = random.randint(0,len(possible_positions) - 1)
    start_scale_idx = random.randint(0, len(possible_scales) - 1)
    start_color_idx = random.randint(0, len(possible_color_indices) - 1)
        
    return make_sequence(sequence_length,
                                     start_prim_index,
                                     start_quadrant,
                                     start_scale_idx,
                                     start_color_idx,
                                     primitive_names,
                                     all_scales,
                                     colors,
                                     possible_positions,
                                     next_primitive_offset,
                                     next_scale_offset,
                                     next_position_offset,
                                     next_color_offset,
                                     noisy_in_quadrant_locations=noisy_in_quadrant_locations)


def get_quadrant_from_location(canvas_size,location):
    x,y = location
    if x < canvas_size/2 and y < canvas_size/2:
        return 0
    elif x >= canvas_size/2 and y < canvas_size/2:
        return 1
    elif x < canvas_size/2 and y >= canvas_size/2:
        return 2
    else:
        return 3
    

def generate_disentanglement_task_metadata(test_latent,
                               all_primitives,
                               all_scales,
                               all_colors,
                               all_positions,
                               canvas_size,
                               train_set,
                               all_contexts=[]):
    
    primitive_names = list(all_primitives.keys())
    all_latent_values = {"primitive": primitive_names,
                         "scale": all_scales,
                         "color": all_colors,
                         "position": all_positions}
    assert test_latent in all_latent_values, f"Test latent '{test_latent}' not found in all_latent_values keys: {list(all_latent_values.keys())}"
    
    half_latent_values = {}
    for key in all_latent_values:
        if key != test_latent:
            if len(all_latent_values[key]) == 1:
                half_latent_values[key] = all_latent_values[key]
            else:
                if train_set:
                    half_latent_values[key] = all_latent_values[key][:len(all_latent_values[key]) // 2]
                else:
                    half_latent_values[key] = all_latent_values[key][len(all_latent_values[key]) // 2:]
        else:
            half_latent_values[key] = all_latent_values[key]
     
    abstract_metadata = []
    for prim_name in half_latent_values['primitive']:
        for scale in half_latent_values['scale']:
            for color in half_latent_values['color']:
                for pos in half_latent_values['position']:                        
                    metadata = {
                        'name': prim_name,
                        'scale': scale,
                        'color': color,
                        'position': pos}
                    abstract_metadata.append([metadata])

    return abstract_metadata

class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 num_drawings=500,
                 sequence_length=4,
                 canvas_size=64,
                 unit_length=12,
                 scales=[1],
                 colors = ["white"],
                 next_primitive_offset=0,
                 next_scale_offset=0,
                 next_position_offset=0,
                 next_color_offset=0,
                 colorize_white=False,
                 premake_videos=False,
                 random_data=True,
                 noisy_in_quadrant_locations=False,
                 task_disentanglement=False,
                 task_disentanglement_contexts=[],
                 task_disentanglement_latent="primitive",
                 task_disentanglement_train_set=True,
                 ): 
        
        self.sequence_length = sequence_length
        self.next_primitive_offset = next_primitive_offset
        self.next_scale_offset = next_scale_offset
        self.next_position_offset = next_position_offset
        self.num_drawings = num_drawings
        self.canvas_size = canvas_size
        self.unit_length = unit_length
        self.scales = scales
        self.all_primitives = self.create_primitives()
        self.primitive_names = list(self.all_primitives.keys())
        print(f"Using primitives: {self.primitive_names}")
        self.premake_videos = premake_videos
        self.final_images = []
        self.primitive_sequences = []
        self.drawings_metadata = []
        self.all_colors = colors
        self.next_color_offset = next_color_offset
        self.colorize_white = colorize_white
        self.randomize_shape_offset_in_sequence_images = False
        assert (not (task_disentanglement and random_data)), "Choose either  task_disentanglement or random data not both."
        q = canvas_size // 4     
        self.positions = [(q,q), (3*q,q), (q,3*q), (3*q,3*q)]  
        self.drawing_lengths = []
        if random_data:
            for _ in range(self.num_drawings):
                metadata = generate_metadata(
                    self.all_primitives,
                    scales,
                    colors,
                    self.positions,
                    canvas_size,
                    sequence_length=self.sequence_length,
                    next_primitive_offset=next_primitive_offset,
                    next_scale_offset=next_scale_offset,
                    next_position_offset=next_position_offset,
                    next_color_offset=next_color_offset,
                    return_image=premake_videos,
                    noisy_in_quadrant_locations=noisy_in_quadrant_locations)
                                
                self.drawings_metadata.append(metadata)
                self.drawing_lengths.append(len(metadata))
            print(f"Generated {len(self.drawings_metadata)} random drawings.")
                    
        elif task_disentanglement:
            self.drawings_metadata = generate_disentanglement_task_metadata(
                task_disentanglement_latent,
                self.all_primitives,
                scales,
                colors,
                self.positions,
                canvas_size,
                train_set=task_disentanglement_train_set,
                all_contexts=task_disentanglement_contexts)
            
            print(f"Generated {len(self.drawings_metadata)} disentanglement task drawings.")
        
    def __len__(self):
        return len(self.drawings_metadata)

    def get_output_dims(self):
        return [len(self.all_primitives), len(self.scales), len(self.all_colors), self.canvas_size+1, self.canvas_size+1]

    def get_output_names(self):
        return ['primitive', 'scale', 'color', 'quadrant']

    def create_primitives(self):
        ul, ul_half = self.unit_length, self.unit_length / 2.0
        shapes_new = {
            'h_line': [{'type': 'line', 'start': (-ul_half, 0), 'end': (ul_half, 0)}],
            'v_line': [{'type': 'line', 'start': (0, -ul_half), 'end': (0, ul_half)}],
            'circle': [create_symmetrical_arc(ul_half, gap_angle=0)],
            'semi_circle': [create_symmetrical_arc(ul_half, gap_angle=SEMI_CIRCLE_START)],
            'cross': [
                {'type': 'line', 'start': (-ul_half, 0), 'end': (ul_half, 0)},           # horizontal
                {'type': 'line', 'start': (0, -ul_half), 'end': (0, ul_half)},           # vertical
            ],
            'triangle': [
                {'type': 'line', 'start': (-ul_half, -ul_half), 'end': (ul_half, -ul_half)},  # base
                {'type': 'line', 'start': (ul_half, -ul_half), 'end': (0, ul_half)},         # right edge
                {'type': 'line', 'start': (0, ul_half), 'end': (-ul_half, -ul_half)},        # left edge
            ],
            'square': [
                {'type': 'line', 'start': (-ul_half, -ul_half), 'end': (ul_half, -ul_half)}, # bottom
                {'type': 'line', 'start': (ul_half, -ul_half), 'end': (ul_half, ul_half)},   # right
                {'type': 'line', 'start': (ul_half, ul_half), 'end': (-ul_half, ul_half)},   # top
                {'type': 'line', 'start': (-ul_half, ul_half), 'end': (-ul_half, -ul_half)}, # left
                ],
        }

        return shapes_new

    def make_prim_sequence(self, metadata,randomize_shape_offset_in_sequence_images):
        
        every_primitive_imgs = []
        offsets = []
        for meta_idx,meta in enumerate(metadata):
            prim_template = self.all_primitives[meta['name']]
            offset = meta['position'] if not randomize_shape_offset_in_sequence_images else (random.randint(self.unit_length, self.canvas_size-self.unit_length), random.randint(self.unit_length, self.canvas_size-self.unit_length))
            placed_primitive = [transform_shape(s, offset, meta['scale']) for s in prim_template]
            shape_color = meta['color']
            prim_img = draw_shapes_on_image(placed_primitive, self.canvas_size, shape_color=shape_color,colorized = len(self.all_colors) > 1)
            prim_tensor = torch.tensor(np.array(prim_img))
            prim_tensor = prim_tensor.float() /  255.0 
            if len(prim_tensor.shape) == 2:
                prim_tensor = prim_tensor.unsqueeze(0)  
            else:
                prim_tensor = prim_tensor.permute(2, 0, 1)
            every_primitive_imgs.append(prim_tensor)
            offsets.append(offset)
        
        every_primitive_img_tensor = torch.stack(every_primitive_imgs, dim=0)
        return every_primitive_img_tensor, offsets
    
    def pad_prim_sequence(self, sequence, max_length):
        if sequence.shape[0] < max_length:
            padding = torch.zeros((max_length - sequence.shape[0], *sequence.shape[1:]), dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)
        return sequence[:max_length]
        
    def __getitem__(self, idx):
        metadata = copy.deepcopy(self.drawings_metadata[idx])
        if not self.premake_videos:
            prim_sequence_input,_  = self.make_prim_sequence(self.drawings_metadata[idx],randomize_shape_offset_in_sequence_images=False)
        else:
            prim_sequence_input = self.primitive_sequences[idx]
        
        labels = torch.tensor([self.primitive_names.index(m['name']) for m in metadata])
        scales = torch.tensor([self.scales.index(m['scale']) for m in metadata])
        colors = torch.tensor([self.all_colors.index(m['color']) for m in metadata])
        position = torch.tensor([self.positions.index(m['position']) for m in metadata], dtype=torch.float32)
        latents = torch.cat([labels.unsqueeze(1),scales.unsqueeze(1),colors.unsqueeze(1),position.unsqueeze(1)], dim=1)
        
        return prim_sequence_input, latents


def make_dataset(num_samples,
                 config,
                 task_disentanglement=False,
                 task_disentanglement_latent="primitive",
                 task_disentanglement_contexts=[],
                 task_disentanglement_train_set=True,
                 next_primitive_offset=0,
                 next_scale_offset=0,
                 next_position_offset=0,
                 next_color_offset=0):
    
    ds = SketchDataset(num_samples,
                    sequence_length=config.dataset.sequence_length,
                    canvas_size=config.dataset.canvas_size,
                    unit_length=config.dataset.unit_length,
                    scales=config.dataset.scales,
                    colors= config.dataset.colors,
                    next_primitive_offset=next_primitive_offset,
                    next_scale_offset=next_scale_offset,
                    next_position_offset=next_position_offset,
                    next_color_offset=next_color_offset,
                    colorize_white=config.dataset.colorize_white,
                    premake_videos=config.dataset.premake_videos,
                    random_data= not  task_disentanglement,
                    task_disentanglement=task_disentanglement,
                    task_disentanglement_latent=task_disentanglement_latent,
                    task_disentanglement_contexts=task_disentanglement_contexts,
                    task_disentanglement_train_set=task_disentanglement_train_set,
                    noisy_in_quadrant_locations=config.dataset.noisy_in_quadrant_locations)    

    return ds


def make_contextual_sketch_collate(device):
    def collate_fn(batch):
        prim_in     = torch.stack([item[0] for item in batch], dim=0)  # [B, T, ...]
        latents     = torch.stack([item[1] for item in batch], dim=0)  # [B, T, ...]
        
        prim_in     = prim_in.transpose(0,1).to(device)
        latents     = latents.transpose(0,1).to(device)
        contexts = torch.stack([item[-1] for item in batch], dim=0).to(device)  # [B, C]
        
        return prim_in, latents, contexts
    return collate_fn


def make_sketch_loader(config,
                    contexts,
                    context_vector_size,
                    context_start_idx=0,
                    num_drawing_per_context=1000,
                    put_in_dict=False):

    if not put_in_dict:
        datasets = []
        context_vectors = []
    else:
        datasets = {}
        
    for ctxt_val_idx,ctxt in enumerate(contexts):
        ctxt_val = list(ctxt.values())
        context_ds = SketchDataset(num_drawing_per_context,
                    sequence_length=config.dataset.sequence_length,
                    canvas_size=config.dataset.canvas_size,
                    unit_length=config.dataset.unit_length,
                    scales=config.dataset.scales,
                    colors=config.dataset.colors,
                    colorize_white=config.dataset.colorize_white,
                    premake_videos=config.dataset.premake_videos,
                    random_data=True,
                    **ctxt)
        if config.one_hot_context:
            context_vector = torch.zeros((context_vector_size))
            context_vector[ctxt_val_idx+context_start_idx] = 1
        else:
            context_vector = torch.tensor(ctxt_val, dtype=torch.float32)
        if put_in_dict:
            context_name = "".join([f"{config.dataset.contexts[i]}_{ctxt_val[i]}" for i in range(len(ctxt_val))])
            concat_ds = (context_ds,context_vector.unsqueeze(0))
            datasets[context_name] = concat_ds
        else:
            datasets.append(context_ds)
            context_vectors.append(context_vector)

    if put_in_dict:
        return datasets, None  
    
    context_vectors = torch.stack(context_vectors)
    print("Contexts: ",context_vectors)
    return datasets, context_vectors


