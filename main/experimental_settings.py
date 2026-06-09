SETTINGS = {
    "experiments": {
        "zeroshot_classification_aircraft": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "Aircraft", "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/fgvc_aircraft.txt"},
        },
        "zeroshot_classification_bird": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "Bird", "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/cub.txt"},
        },
        "zeroshot_classification_caltech101": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "Caltech101",
                            "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/caltech101.txt"},
        },
        "zeroshot_classification_car": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "Car", "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/stanford_cars.txt"},
        },
        
        "zeroshot_classification_dtd_fold1": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 1}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold2": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 2}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold3": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 3}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold4": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 4}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold5": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 5}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold6": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 6}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold7": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 7}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold8": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 8}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold9": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 9}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_dtd_fold10": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "DTD", "args": {"test": True, "fold": 10}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/dtd.txt"},
        },
        "zeroshot_classification_eurosat": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "EuroSAT", "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/eurosat.txt"},
        },
        "zeroshot_classification_flower": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "Flower", "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/oxford_flowers.txt"},
        },
        "zeroshot_classification_food": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "Food", "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/food101.txt"},
        },


        "zeroshot_classification_imagenet": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "ImageNet", "args": {"test": True}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/imagenet.txt"},
        },


        "zeroshot_classification_pet": {
            "test_function": "autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "autodl-tmp/data/generic.py", "dataset_name": "Pet", "args": {"test": True}},
            "test_args": {"classname_file": "autodl-tmp/data/classnames/oxford_pets.txt"},
        },
        "zeroshot_classification_sun_fold1": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 1}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold2": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 2}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold3": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 3}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold4": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 4}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold5": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 5}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold6": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 6}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold7": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 7}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold8": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 8}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold9": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 9}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        "zeroshot_classification_sun_fold10": {
            "test_function": "/root/autodl-tmp/metric/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "/root/autodl-tmp/data/generic.py", "dataset_name": "SUN397", "args": {"test": True, "fold": 10}},
            "test_args": {"classname_file": "/root/autodl-tmp/data/classnames/sun397.txt"},
        },
        # "zeroshot_classification_ucf": {
        #     "test_function": "metric/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "UCF101", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/ucf101.txt"},
        # },
        # "zeroshot_retrieval_flicker8k": {
        #     "test_function": "/root/autodl-tmp/metric/zeroshot_retrieval.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {
        #         "dataset_func": "/root/autodl-tmp/data/flickr.py",
        #         "dataset_name": "Flickr8K",
        #         "args": {"tokenizer": None, "transform": None, "test": True},
        #     },
        # },
        # "zeroshot_retrieval_flicker30k": {
        #     "test_function": "metric/zeroshot_retrieval.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {
        #         "dataset_func": "data/flickr.py",
        #         "dataset_name": "Flickr30K",
        #         "args": {"tokenizer": None, "transform": None, "test": True},
        #     },
        # },
    }
}

FEAT_VIZ_SETTINGS = {
    "dataset_cls": {
        "dataset_func": "/root/autodl-tmp/data/flickr.py",
        "dataset_name": "Flickr8K",
        "args": {"tokenizer": None, "transform": None, "test": True},
    },
}
