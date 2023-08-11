from noise_api.real_noise import real_noise, real_noise_config

# Fully customized Noise Injection.
# real_noise(
#         in_file="noise_api/examples/test.mp4",
#         out_file="noise_api/examples/test_out.mp4",
#         mode="exact",
#         v_mode=["avgblur"],
#         v_start=[1.0],
#         v_end=[2.0],
#         v_option=[(2,2)],
#         a_mode=["coloran", "coloran", "lowpass", "reverb", "coloran"],
#         a_start=[0.0, 2.0, 4.0, 6.0, 7.0],
#         a_end=[2.0, 4.0, 6.0, 7.0, 9.0],
#         a_option=[('pink', 0.8), ('brown', 1.0), 500, ("room", 1.0), ('white', 0.8)],
#     )

# Random Noise Injection. Used for quantitative evaluation.
cfg = real_noise_config(
    "noise_api/examples/test.mp4", 
    mode = "random_full", 
    v_noise_list=["avgblur", "impulse_value"], 
    v_noise_num = 2,
    v_noise_ratio = 0.8,
    v_noise_intensity = 0.5,
    a_noise_list = ["reverb"],
    a_noise_num = 1,
    a_noise_ratio = 1.0,
    a_noise_intensity = 1,
)._asdict()
print(cfg)
real_noise("noise_api/examples/test.mp4", "noise_api/examples/test_out.mp4", **cfg)