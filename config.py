"""
This file is the configuration of our network on various datasets
"""
def get_covid_config():
    return {
        'patch_size': (224, 224, 28),
        'res_ratio': 5/0.74,  # res_ratio = resolution xy axis / resolution z axis
        'layers': (15,4),
        'in_chans': 1,
        'num_classes': 2,
        'embed_dims': (42, 84, 168, 168, 336),
        'segment_dim': (14,7),
        'mlp_ratio' : 4.0, 
        'dropout_rate' : 0.2
    }


def get_synapse_config():
    return {
        'patch_size': (192, 192, 48),  # the patch size and resolution is automatically determined by nnUNet preprocessing.
        'res_ratio': 4.0,
        'layers': (15,4),
        'in_chans': 1,
        'num_classes': 9,
        'embed_dims': (48, 96, 192, 192, 384),
        'segment_dim': (12,6),
        'mlp_ratio' : 4.0, 
        'dropout_rate' : 0.3
    }


def get_brats_config():
    return {
        'patch_size': (128, 128, 128),  
        'res_ratio': 1.0,
        'layers': (15,4),
        'in_chans': 4,
        'num_classes': 4,
        'embed_dims': (48, 96, 192, 192, 384),
        'segment_dim': (8,4),
        'mlp_ratio' : 4.0, 
        'dropout_rate' : 0.3
    }


def get_lits_config():
    return {
        'patch_size': (96, 96, 96),  
        'res_ratio': 1.0,
        'layers': (15,4),
        'in_chans': 1,
        'num_classes': 2,
        'embed_dims': (48, 96, 192, 192, 384),
        'segment_dim': (6, 3),
        'mlp_ratio' : 4.0, 
        'dropout_rate' : 0.3
    }