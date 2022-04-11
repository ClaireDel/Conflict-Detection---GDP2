import random as rd

FRAMERATE = 30
VIOLENCE_THRESHOLD = 6
def dist(pos1, pos2):
    return ((abs(pos1[0]-pos2[0]))**2 + (abs(pos1[1]-pos2[1]))**2)**0.5

def rectangle_surface(start_point, end_point):
    return(end_point[0]-start_point[0])*(end_point[1]-start_point[1])

def get_overlap_rectangle(box0, box1):
    r0_spt, r0_ept = box0[0], box0[1]
    r1_spt, r1_ept = box1[0], box1[1]
    overlap_spt = max((r0_spt[0], r1_spt[0])), max((r0_spt[1], r1_spt[1]))
    overlap_ept = min((r0_ept[0], r1_ept[0])), min((r0_ept[1], r1_ept[1]))
    return (overlap_spt, overlap_ept)
def overlapping_score(box0, box1):
    overlap_spt, overlap_ept = get_overlap_rectangle(box0,box1)
    overlap_surface = rectangle_surface(overlap_spt, overlap_ept)
    box0_surface = rectangle_surface(box0[0], box0[1])
    return overlap_surface/box0_surface

class Person():
##DELETING INSTANCES WHEN THEY ARE NOT IN THE FRAME ANYMORE
    threshold = 0.80
    poses_cache_size = 30
    instance_count = 0
    skeleton_fields = [
        'img',
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
        ]

    def __init__(self, position=(0,0), box=((0,0), (0,0))) -> None:
        self.id = rd.randint(0, 2000)
        Person.instance_count += 1
        self.poses = []
        self.position = position
        self.bounding_box = box

    def is_you(self, box):
        return overlapping_score(self.bounding_box, box) > Person.threshold

    def add_pose(self, pose):
        if len(self.poses) < Person.poses_cache_size:
            self.poses.append(pose)
        else:
            self.poses.pop(0)
            self.poses.append(pose)
    
    def update_position(self, position, box):
        self.position = position
        self.bounding_box = box
    
    def velocity(self):
        left_wrist_positions = [self.poses[k][9] for k in range(len(self.poses))]
        right_wrist_positions = [self.poses[k][10] for k in range(len(self.poses))]
        left_ankle_positions = [self.poses[k][15] for k in range(len(self.poses))]
        right_ankle_positions = [self.poses[k][16] for k in range(len(self.poses))]
        vel_lhand, vel_rhand, vel_lfoot, vel_rfoot = 0, 0, 0, 0
        if len(self.poses)>3:
            vel_lhand = dist(left_wrist_positions[-1], left_wrist_positions[-2])*FRAMERATE
            vel_rhand = dist(right_wrist_positions[-1], right_wrist_positions[-2])*FRAMERATE
            vel_lfoot = dist(left_ankle_positions[-1], left_ankle_positions[-2])*FRAMERATE
            vel_rfoot = dist(right_ankle_positions[-1], right_ankle_positions[-2])*FRAMERATE


        return vel_lhand, vel_rhand, vel_lfoot, vel_rfoot

    def aggressive(self):
        vel_lhand, vel_rhand, vel_lfoot, vel_rfoot = self.velocity()
        return vel_lhand> VIOLENCE_THRESHOLD or vel_rhand> VIOLENCE_THRESHOLD or vel_rfoot> VIOLENCE_THRESHOLD or vel_lfoot> VIOLENCE_THRESHOLD

        
