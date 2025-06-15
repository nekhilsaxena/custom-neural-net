#include "Visualization.h"
#include "imgui.h"
#include "implot.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include "../Utils/Utils.h"

static GLFWwindow *g_Window = nullptr;

void setVisualizationWindow(GLFWwindow *window)
{
     g_Window = window;
}

void Visualization::initialize()
{
     if (!g_Window)
     {
          throw std::runtime_error("Visualization error: GLFW window not set. Call setVisualizationWindow() first.");
     }

     IMGUI_CHECKVERSION();
     ImGui::CreateContext();
     ImPlot::CreateContext();

     ImGuiIO &io = ImGui::GetIO();
     (void)io;
     io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

     ImGui::StyleColorsDark();

     ImGui_ImplGlfw_InitForOpenGL(g_Window, true);
     ImGui_ImplOpenGL3_Init("#version 130");

     std::cout << "Visualization initialized.\n";
}

void Visualization::shutdown()
{
     ImGui_ImplOpenGL3_Shutdown();
     ImGui_ImplGlfw_Shutdown();
     ImPlot::DestroyContext();
     ImGui::DestroyContext();
     std::cout << "Visualization shutdown.\n";
}

void Visualization::beginFrame()
{
     glfwPollEvents();
     ImGui_ImplOpenGL3_NewFrame();
     ImGui_ImplGlfw_NewFrame();
     ImGui::NewFrame();
}

void Visualization::endFrame()
{
     ImGui::Render();
     int display_w, display_h;
     glfwGetFramebufferSize(g_Window, &display_w, &display_h);
     glViewport(0, 0, display_w, display_h);
     glClear(GL_COLOR_BUFFER_BIT);
     ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
     glfwSwapBuffers(g_Window);
}

void Visualization::updateCost(float epoch, float cost)
{
     epochHistory.push_back(epoch);
     costHistory.push_back(cost);
}

void Visualization::render()
{
     ImGui::SetNextWindowSize(ImVec2(1000, 600), ImGuiCond_Once);
     ImGui::Begin("Training Visualization");

     ImVec2 plotSize(900, 500);
     if (ImPlot::BeginPlot("Cost Over Epochs", "Epoch", "Cost", plotSize))
     {
          if (!epochHistory.empty() && !costHistory.empty())
          {
               StatsSummary stats = computeStatistics(costHistory);
               float upperLimit = stats.max + (1.5f * (stats.q3 - stats.q1));

               ImPlot::SetupAxesLimits(
                   epochHistory.front(),
                   epochHistory.back(),
                   0.0f,
                   upperLimit);

               ImPlot::PlotLine("Cost", epochHistory.data(), costHistory.data(), (int)costHistory.size());
          }

          ImPlot::EndPlot();
     }

     ImGui::End();
}
