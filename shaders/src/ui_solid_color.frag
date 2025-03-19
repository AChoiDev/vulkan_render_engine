#version 460

layout(location = 0) out vec4 finalColor;

layout( push_constant ) uniform UIQuadJob {
    vec4 color;
    vec4 screen_res;
} job;

void main() {
    finalColor = job.color;
}