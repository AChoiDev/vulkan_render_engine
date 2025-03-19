#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform MyUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} mubo;

layout(set = 0, binding = 1) uniform sampler texSampler;

struct MySSBObject {
    mat4 model;
    uvec4 data_one;
};

struct TextureGroupDesc {
    uint base_color;
    uint normal;
    uint emissive;
    uint metallic;

    uint roughness;
    uint occlusion;
    uint padding_a;
    uint padding_b;
};

layout(std140, set = 0, binding = 2) readonly buffer MySSBuffer {
    MySSBObject obj[];
} mssb;

// TODO: nonuniform qualifier for variable sized array
layout(set = 0, binding = 3) uniform texture2D texArray[1024];

layout(std140, set = 0, binding = 4) readonly buffer MyTexGroupBuffer {
    TextureGroupDesc obj[];
} tex_group_desc_buffer;

// Vertex Input Attributes
layout(location = 0) in vec3 oColor;
layout(location = 1) in vec2 oTexCoord;
layout(location = 2) in vec3 interp_vertex_normal;
layout(location = 3) flat in uint i_gl_InstanceID;
layout(location = 4) in vec3 interp_vertex_tangent;
layout(location = 5) in smooth mat3 fragTBN_view_space;

// Output
layout(location = 0) out vec4 finalColor;

void main() {
    finalColor = vec4(0.6, 0.3, 0.2, 1.0);
}
