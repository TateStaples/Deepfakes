# actions
from first_order_model.train import train

# normal modules
import yaml, os, imageio
from first_order_model.demo import find_best_frame, resize, load_checkpoints, make_animation, img_as_ubyte

# networks
from first_order_model.modules.discriminator import MultiScaleDiscriminator
from first_order_model.modules.generator import OcclusionAwareGenerator
from first_order_model.modules.keypoint_detector import KPDetector
from first_order_model.frames_dataset import FramesDataset


def get_video(target_img_path, driving_video_path):  # based off run
    config = yaml.load(open("first_order_model/config/vox-256.yaml"))
    mode = "animate"
    log_dir = "video"
    checkpoint = None
    device_ids = "0"
    verbose = False

    log_dir = os.path.join(log_dir, os.path.basename(config).split('.')[0])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    dataset = FramesDataset(is_train=(mode == 'train'), **config['dataset_params'])
    train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids)


def demo_video(path_to_img, path_to_video, output_file, auto_crop=True):
    config = "first_order_model/config/vox-256.yaml"  # model settings
    checkpoint = "first_order_model/vox-cpk.pth.tar"  # actual model
    cpu = True  # using cpu not gpu
    relative = False  # make relative motions or move to absolute location
    auto_crop = auto_crop
    adapt_scale = True
    result_video = output_file
    best_frame = None  # where to start

    source_image = imageio.imread(path_to_img)
    reader = imageio.get_reader(path_to_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=config, checkpoint_path=checkpoint, cpu=cpu)

    if auto_crop or best_frame is not None:
        i = best_frame if best_frame is not None else find_best_frame(source_image, driving_video, cpu=cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=cpu)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=cpu)
    video = [img_as_ubyte(frame) for frame in predictions]
    if output_file is not None:
        imageio.mimsave(result_video, video, fps=fps)
    return video, fps
