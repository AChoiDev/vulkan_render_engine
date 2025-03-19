#version 460

layout(set = 0, binding = 1) uniform sampler texSampler;
layout(set = 0, binding = 3) uniform texture2D texArray[1024];

layout(location = 1) in vec2 oTexCoord;
layout(location = 2) in vec2 oQuadSize;

layout(location = 0) out vec4 finalColor;

layout( push_constant ) uniform UIQuadJob {
    vec4 color;
    vec4 resolution;
    uvec4 indices;
} job;

void main() {
    uint tex_index = job.indices.x;
    float signed_dist = texture(sampler2D(texArray[tex_index], texSampler), oTexCoord).r * -2.0 + 1.0;
    float dist_radius = 7.0 * (1.0 / job.resolution.x) * (1.0 / oQuadSize.x);
    // float dist_radius = ((resolution_width / ui_width) * glyph_width_u_width) * 0.5;
    // float dist_radius = 0.2;

    if (signed_dist > -dist_radius && signed_dist < dist_radius) {
        float norm_t = 1.0 - (signed_dist + dist_radius) / (2.0 * dist_radius);
        float alpha = smoothstep(0.0, 1.0, norm_t);
        // float alpha = norm_t;
        finalColor = vec4(job.color.rgb, alpha * job.color.a);
        // finalColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
    else if (signed_dist < -dist_radius)  {
        finalColor = job.color;
    }
    else {
        finalColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}