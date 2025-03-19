use std::collections::HashMap;

use crate::{PrefabAnimationID, PrefabNodeID, RuntimeNodeID, RuntimeScene};
pub struct MyPrefabAnimChannel {
    pub target: PrefabNodeID,
    pub interpolation: MyAnimInterpolation,
    pub inputs: Vec<f32>,
    pub outputs: MyAnimChannelOutputList,
}

#[derive(Debug, Clone, Copy)]
pub enum MyAnimInterpolation {
    Linear,
    Step,
    CubicSpline,
}

#[derive(Debug, Clone)]
pub enum MyAnimChannelOutputList {
    Translation(Vec<glam::Vec3>),
    Rotation(Vec<glam::Quat>),
    Scale(Vec<glam::Vec3>),
    MorphTargetWeight(Vec<f32>),
}

pub struct MyPrefabAnimation {
    pub channels: Vec<MyPrefabAnimChannel>,
    pub name: Option<String>,
}

pub struct AnimationBinding {
    pub channel_node_bindings: Vec<RuntimeNodeID>,
    pub animation_id: PrefabAnimationID,
}

trait AnimationOutput: Sized + Clone {
    fn lerp(self, other: Self, f: f32) -> Self;
}

impl AnimationOutput for glam::Vec3 {
    fn lerp(self, other: Self, f: f32) -> Self {
        self.lerp(other, f)
    }
}
impl AnimationOutput for glam::Quat {
    fn lerp(self, other: Self, f: f32) -> Self {
        self.lerp(other, f)
    }
}

// find two keyframes to interpolate between
// these may be the same keyframe in some cases
// inputs: monotonically strictly increasing, all values non-negative
fn interpolate<T: AnimationOutput>(
    target_time: f32,
    inputs: &Vec<f32>,
    outputs: &Vec<T>,
    interpolation: MyAnimInterpolation,
) -> T {
    assert!(inputs.len() > 0);
    // Find a keyframe closest to the target time, preferring that it precedes the target
    let mut left_idx: usize = 0;
    for (idx, input) in inputs.iter().enumerate() {
        if *input <= target_time {
            left_idx = idx;
        } else {
            break;
        }
    }
    let left_output = outputs[left_idx].clone();

    match interpolation {
        MyAnimInterpolation::Linear => {
            // Find the keyframe for the right-endpoint of the interpolation
            let left_input = inputs[left_idx];
            let right_idx = if left_input == target_time {
                // left input matches target time exactly
                left_idx
            } else {
                if inputs.get(left_idx + 1) != None {
                    // there exists an input that comes after the left-most input
                    left_idx + 1
                } else {
                    // no inputs exist past the left input, use left input
                    left_idx
                }
            };
            let right_input = inputs[right_idx];
            let right_output = outputs[right_idx].clone();


            // apply linear interpolation
            let clamped_time = target_time.clamp(left_input, right_input);
            let segment_duration = right_input - left_input;

            const MINIMUM_INTERP_SEGMENT_LENGTH: f32 = f32::EPSILON * 8.0;
            if segment_duration >= MINIMUM_INTERP_SEGMENT_LENGTH {
                let factor = (clamped_time - left_input) / segment_duration;
                left_output.lerp(right_output, factor)
            } else {
                // segment duration is too small, use left output
                left_output
            }
        }
        MyAnimInterpolation::Step => {
            left_output
        },
        MyAnimInterpolation::CubicSpline => todo!(),
    }
}

impl AnimationBinding {
    pub fn apply(&self, runtime_scene: &mut RuntimeScene, target_time: f32, prefab_animations: &HashMap::<PrefabAnimationID, MyPrefabAnimation>) {
        let prefab_animation = &prefab_animations[&self.animation_id]; // TODO: make this more safe
        for (channel_idx, channel_node_binding) in self.channel_node_bindings.iter().enumerate() {
            let channel = &prefab_animation.channels[channel_idx]; // TODO: make this more safe

            // channel has no keyframes, skip this channel
            if channel.inputs.len() == 0 {
                continue;
            }

            // if target node not present, skip this channel
            let target_node = if let Some(node) = runtime_scene.get_mut(*channel_node_binding) {
                node
            } else {
                continue;
            };

            match &channel.outputs {
                MyAnimChannelOutputList::Translation(translations) => {
                    let output = interpolate(target_time, &channel.inputs, translations, channel.interpolation);

                    let (scale, rot, _) =
                        target_node.local_transform.to_scale_rotation_translation();
                    target_node.local_transform =
                        glam::Mat4::from_scale_rotation_translation(scale, rot, output);
                }
                MyAnimChannelOutputList::Rotation(rotations) => {
                    let output = interpolate(target_time, &channel.inputs, rotations, channel.interpolation);

                    let (scale, _, translation) =
                        target_node.local_transform.to_scale_rotation_translation();
                    target_node.local_transform =
                        glam::Mat4::from_scale_rotation_translation(scale, output, translation);
                }
                MyAnimChannelOutputList::Scale(scales) => {
                    let output = interpolate(target_time, &channel.inputs, scales, channel.interpolation);
                    
                    let (_, rot, translation) =
                        target_node.local_transform.to_scale_rotation_translation();
                    target_node.local_transform =
                        glam::Mat4::from_scale_rotation_translation(output, rot, translation);
                }
                MyAnimChannelOutputList::MorphTargetWeight(_) => {
                    todo!("Weights not implemented");
                },
            }
        }
    }
}
