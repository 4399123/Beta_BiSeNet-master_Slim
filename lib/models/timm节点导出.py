import torch
import timm
x = torch.randn(1, 3, 224, 224)
all_feature_extractor = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', features_only=True)
all_features = all_feature_extractor(x)
print('All {} Features: '.format(len(all_features)))
for i in range(len(all_features)):
    print('feature {} shape: {}'.format(i, all_features[i].shape))