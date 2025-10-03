import torch, os
from desmume.emulator import DeSmuME

from utils.emulator import init_desmume_with_overlay, draw_checkpoints, draw_collisions
from utils.course import checkpoint_data_tensor_3d
from utils.kcl_torch import KCLTensor
from utils.vector import get_mps_device, project_to_screen

from utils.racer import Racer


device = get_mps_device()

kcl_path = "test_course_data/desert/course_collision.kcl"
nkm_path = "test_course_data/desert/course_map.nkm"
racer = Racer.from_path(kcl_path, nkm_path, device=device)
print(racer)
checkpoint1 = checkpoint_data_tensor_3d(
    racer.nkm._CPOI.position1, 
    racer.kcl.positions,
    device=racer.device
)
checkpoint2 = checkpoint_data_tensor_3d(
    racer.nkm._CPOI.position2, 
    racer.kcl.positions,
    device=racer.device
)
checkpoint = torch.cat([
    checkpoint1, 
    checkpoint2
], dim=0)

near = 0.0
far = 20.0

def display_triangles(indices: torch.Tensor, proj, color=(1, 0, 0)):
    global racer, far, near
    if indices.shape[0] == 0:
        return
    tri_pts1, tri_pts2, tri_pts3 = racer.kcl.triangles[indices].chunk(3, dim=1)
    proj_tri_pts1 = proj(tri_pts1.squeeze(1), False)
    proj_tri_pts2 = proj(tri_pts2.squeeze(1), False)
    proj_tri_pts3 = proj(tri_pts3.squeeze(1), False)
    
    
    valid = lambda x: (x[:, 2] > near) & (x[:, 2] < far)
    valid_mask = valid(proj_tri_pts1) & valid(proj_tri_pts2) & valid(proj_tri_pts3)
    proj_tri_pts1 = proj_tri_pts1[valid_mask]
    proj_tri_pts2 = proj_tri_pts2[valid_mask]
    proj_tri_pts3 = proj_tri_pts3[valid_mask]
    
    draw_collisions(
        proj_tri_pts1.tolist(), 
        proj_tri_pts2.tolist(), 
        proj_tri_pts3.tolist(), 
        color=color
    )

def main(emu: DeSmuME):
    global racer
    
    #os.system("clear")
    racer.memory = emu.memory.unsigned
    
    proj = lambda x, clip: project_to_screen(
        x,
        racer.model_view, 
        *racer.camera_fov,
        far,
        near,
        device=racer.device,
        z_clip=clip
    )
    
    # Update checkpoint overlay
    proj_checkpoint = proj(checkpoint, True)
    draw_checkpoints(proj_checkpoint.tolist(), color=(1, 0, 0))
    
    # Update collision triangle overlay
    indices = torch.tensor(racer.kcl.search_triangles(racer.position), dtype=torch.int32, device=device)
    
    color_map = [
        (racer.kcl.prisms.is_wall, (1, 0, 1), 1),
        (racer.kcl.prisms.is_floor, (0, 0.5, 1), 1),
        #(racer.kcl.prisms.collision_type, (1, 0, 0.3), 5),
    ]
    
    for attr, color, cond in color_map:
        indices_mask = attr[indices] == cond
        display_triangles(indices[indices_mask], proj, color=color)
        
    
    

if __name__ == "__main__":
    init_desmume_with_overlay("mariokart_ds.nds", main)
    
