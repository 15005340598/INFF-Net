# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001,
                 #paramwise_cfg=dict(
                  #   custom_keys={
                  #       # 匹配所有参数名中包含 ARConv 模块的参数（conv1/conv2 用了 ARConv）
                         # 规则：参数名包含 'conv1'/'conv2' 且属于 ARConv 模块，或直接匹配 'ARConv' 特征
                #         'conv1': dict(lr_mult=1.0),  # BasicBlock/Bottleneck 中第一个卷积（ARConv）
                    #     'conv2': dict(lr_mult=1.0),  # BasicBlock/Bottleneck 中第二个卷积（ARConv）
                         # 匹配所有GECA模块（geca/geca1/geca3）
                   #      'geca': dict(lr_mult=1.0),
                      #   'geca1': dict(lr_mult=1.0),
                      #   'geca3': dict(lr_mult=1.0)
                  #   }
            #     )
                 )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
