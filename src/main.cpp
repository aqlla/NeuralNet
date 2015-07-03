#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/Buffer.h>
#include <Magnum/Mesh.h>

#include <iostream>
#include "neuralnet.h"

using namespace Magnum;

struct Vert {
    Vector3 position;
    Color3 color;
};

class Tank {
    constexpr static const float scale = .3f;
    constexpr static const float size = 0.5f;
    constexpr static const float gunLength = 0.5f;
    constexpr static const float gunWidth = 0.1f;

public:
    Mesh mesh;
    Buffer vertexBuffer;

    float x;
    float y;

    Tank(float x, float y) : x{x}, y{y} {
        Vert data[] = {
                {{-size * scale, -size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{ size * scale, -size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{ size * scale,  size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{-size * scale, -size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{ size * scale,  size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{-size * scale,  size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},

                {{-gunWidth * scale, size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{ gunWidth * scale, size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{ gunWidth * scale, (size+gunLength) * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{-gunWidth * scale, size * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{ gunWidth * scale, (size+gunLength) * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
                {{-gunWidth * scale, (size+gunLength) * scale, 0.0f}, {1.0f, 0.0f, 0.0f}},
        };

        vertexBuffer.setData(data, BufferUsage::StaticDraw);
        mesh.setPrimitive(MeshPrimitive::Triangles)
            .setCount(12)
            .addVertexBuffer(vertexBuffer, 0, Shaders::VertexColor3D::Position{}, Shaders::VertexColor3D::Color{});
    };


};

class NeuralApp : public Platform::Application
{
public:
    explicit NeuralApp(const Arguments &arguments);

private:
    void drawEvent() override;
    void testNet();
    Tank *t;

    constexpr static const int viewWidth = 600;
    constexpr static const int viewHeight = 600;
};

NeuralApp::NeuralApp(const Arguments &arguments)
        : Platform::Application{arguments, Configuration{}.setSize({NeuralApp::viewWidth, NeuralApp::viewHeight}).setTitle("Neural Net")}, t{new Tank{.75, .75}}
{ }

void NeuralApp::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color);

    Shaders::VertexColor3D shader;
//    Matrix4 transformationMatrix = Matrix4::translation(Vector3::zAxis(-5.0f));
//    Matrix4 projectionMatrix = Matrix4::perspectiveProjection(35.0_degf, 1.0f, 0.001f, 100.0f);
//    shader.setTransformationProjectionMatrix(transformationMatrix);
    t->mesh.draw(shader);

    swapBuffers();
}

MAGNUM_APPLICATION_MAIN(NeuralApp)

void NeuralApp::testNet() {
    NeuralNet net;
    net << InputLayer{2}
        << FullyConnectedLayer<activation::sigmoid>{3};
    net.forward(.7, .44);
}
