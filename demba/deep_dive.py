from pathlib import Path
import cv2
import deeplabcut.pose_estimation_pytorch as pep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

config_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/config.yaml'
video_path = '/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/BHVE_group9_316800-318599.mp4'
image_paths = list(Path('/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/test_images').glob('*.png'))
shuffle=3
max_individuals=2
track_method = 'ellipse'

video_path = Path(video_path)
loader = pep.data.DLCLoader(config_path, trainset_index=0, shuffle=shuffle, modelprefix="",
)
pose_cfg_path = loader.model_folder.parent / "test" / "pose_cfg.yaml"
pose_snapshot = pep.apis.utils.get_model_snapshots(-1, loader.model_folder, loader.pose_task)[0]
detector_snapshot = pep.apis.utils.get_model_snapshots(-1, loader.model_folder, pep.task.Task.DETECT)[0]


pose_runner = pep.get_pose_inference_runner(model_config=loader.model_cfg,
                                       snapshot_path=pose_snapshot.path,
                                       max_individuals=max_individuals,)
pose_runner.async_mode = False
detector_runner = pep.apis.utils.get_detector_inference_runner(model_config=loader.model_cfg,
                                                               snapshot_path=detector_snapshot.path,
                                                               max_individuals=max_individuals)

detector_results = detector_runner.inference(image_paths)
pose_results = pose_runner.inference(list(zip(image_paths, detector_results)))

# Create visualizations
def create_visualizations():
    # Save to the same directory as the input images
    output_dir = Path('/home/tlancaster/PycharmProjects/DemBA/projects/demasoni_singlenuc-tucker-2025-09-11/testing/test_images')
    
    # Define colors for different individuals
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, (image_path, pose_result) in enumerate(zip(image_paths, pose_results)):
        # Load the original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot image
        ax.imshow(image)
        ax.set_title(f'Pose Estimation with Bounding Boxes - {image_path.name}')
        ax.axis('off')
        
        # Draw bounding boxes with confidence scores
        if 'bboxes' in pose_result and 'bbox_scores' in pose_result:
            bboxes = pose_result['bboxes']
            bbox_scores = pose_result['bbox_scores']
            
            for i, (bbox, score) in enumerate(zip(bboxes, bbox_scores)):
                x, y, w, h = bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                       edgecolor=colors[i % len(colors)], facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y-10, f'Individual {i+1}\nScore: {score:.3f}', 
                        color=colors[i % len(colors)], fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Draw pose keypoints (without skeletal connections)
        if 'bodyparts' in pose_result:
            bodyparts = pose_result['bodyparts']
            
            for individual_idx, individual_poses in enumerate(bodyparts):
                color = colors[individual_idx % len(colors)]
                
                # Plot keypoints
                valid_points = individual_poses[individual_poses[:, 2] > 0.5]  # confidence threshold
                if len(valid_points) > 0:
                    ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                              c=color, s=50, alpha=0.8, edgecolors='white', linewidth=1)
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = output_dir / f'pose_visualization_{idx:03d}_{image_path.stem}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {output_path}")
    
    print(f"\nAll visualizations saved to: {output_dir}")

# Create the visualizations
create_visualizations()

# video = pep.apis.videos.VideoIterator(str(video_path))
# video_predictions = pep.video_inference(str(video_path), runner, detector_runner)


