#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform MyUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} mubo;

struct MySSBObject {
    mat4 model;
    uvec4 data_one;
};

layout(std140, set = 0, binding = 2) readonly buffer MySSBuffer {
    MySSBObject obj[];
} mssb;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoords;
layout(location = 2) in vec3 inNormals;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 oColor;
layout(location = 1) out vec2 oTexCoord;
layout(location = 2) out vec3 oNormals;
layout(location = 3) out uint o_gl_InstanceIndex;
layout(location = 4) out vec3 oTangent;
layout(location = 5) out mat3 oTBN_view_space;

void main() {
    mat4 model = mssb.obj[gl_InstanceIndex].model;
    gl_Position = mubo.proj * mubo.view * model * vec4(inPosition, 1.0);

    oColor = vec3(1.0, 1.0, 1.0);

    oTexCoord = inTexCoords;
    oNormals = inNormals;
    oTangent = inTangent.xyz;
    o_gl_InstanceIndex = gl_InstanceIndex;

    vec3 bitangent = cross(inNormals, inTangent.xyz) * inTangent.w;

    oTBN_view_space = mat3(mubo.view * model) * mat3(inTangent.xyz, bitangent, inNormals);
}
