#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <iostream>
#include <glm/glm.hpp>
#include <vector>

float pi = 3.14159f;
void grayscale(std::string filepath){
    int width, height, comps;
    int req_comps = 4;
    unsigned char * buffer = stbi_load(filepath.c_str(), &width, &height, &comps, req_comps);
    unsigned char* new_buffer = new unsigned char[width * height]{0};
    for (int i = 0; i < width * height * comps; i = i + comps)
    {
        new_buffer[i/4] = buffer[i]*0.2989 + buffer[i+1]*0.5870 + buffer[i+2]*0.1140;
    }
    int result = stbi_write_png("res/textures/greyscale.png", width, height, 1, new_buffer, width);
    std::cout << "grayscale " << (result ? "success" : "fail") << std::endl;
}

std::vector<float> noise_reduction3x3(std::string oldfilepath, std::string newfilepath){
    int width, height, comps;
    int req_comps = 1;
    unsigned char * buffer = stbi_load(oldfilepath.c_str(), &width, &height, &comps, req_comps);
    unsigned char* new_buffer = new unsigned char[width * height]{0};
    std::vector<float> noise_reduced(width * height);
    const glm::mat3 gaussian3x3 = glm::mat3(
        1.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 1.0f
    ) / 16.0f;
    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){
            float sum = 0.0f;
            for(int ky = -1; ky <= 1; ky++){
                for(int kx = -1; kx <= 1; kx++){
                    int pixelValue = buffer[(y + ky) * width + (x + kx)];
                    sum += pixelValue * gaussian3x3[ky + 1][kx + 1];
                }
            }
            new_buffer[y * width + x] = static_cast<unsigned char>(sum);
            noise_reduced[y * width + x] = sum;
        }
    }
    int result = stbi_write_png(newfilepath.c_str(), width, height, req_comps, new_buffer, width);
    std::cout << "noise reduction " << (result ? "success" : "fail") << std::endl;
    return noise_reduced;
}

std::vector<float> gradiant_intensity(std::string filepath, std::vector<float> noise_reduced, std::vector<float>* gradiant_angle_buffer, int width, int height, int comps){
    int req_comps = 1;
    unsigned char* new_buffer = new unsigned char[width * height]{0};
    std::vector<float> gradiant_magnitude(width * height);
    const glm::mat3 sobelMatX = glm::mat3(
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f
    );
    const glm::mat3 sobelMatY = glm::mat3(
        1.0f, 2.0f, 1.0f,
        0.0f, 0.0f, 0.0f,
        -1.0f, -2.0f, -1.0f
    );
    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){
            float gx = 0.0f;
            float gy = 0.0f;
            for(int ky = -1; ky <= 1; ky++){
                for(int kx = -1; kx <= 1; kx++){
                    int px = glm::clamp(x + kx, 1, width - 2);
                    int py = glm::clamp(y + ky, 1, height - 2);
                    int pixelValue = noise_reduced[py * width + px];
                    gx += pixelValue * sobelMatX[ky + 1][kx + 1];
                    gy += pixelValue * sobelMatY[ky + 1][kx + 1];
                }
            }
            float magnitude = sqrt((gx * gx) + (gy * gy));
            gradiant_magnitude[y * width + x] = magnitude;
            new_buffer[y * width + x] = static_cast<unsigned char>(glm::clamp(magnitude, 0.0f, 255.0f));
            (*gradiant_angle_buffer)[y * width + x] = static_cast<float>(atan2(gy, gx));
        }
    }
    int result = stbi_write_png(filepath.c_str(), width, height, req_comps, new_buffer, width);
    std::cout << "gradient intensity " << (result ? "success" : "fail") << std::endl;
    return gradiant_magnitude;
}

std::vector<float> non_max_suppression(std::string filepath, std::vector<float> gradiant_magnitude, std::vector<float>* gradiant_angle_buffer, int width, int height, int comps){
    int req_comps = 1;
    unsigned char* new_buffer = new unsigned char[width * height]{0};
    std::vector<float> non_max_suppressed(width * height);
    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){
            float angle = (*gradiant_angle_buffer)[y * width + x];
            float magnitude = gradiant_magnitude[y * width + x];
            angle = fmod((angle * 180.0f / pi) + 180.0f, 180.0f);
            float neighbor1 = 0.0f;
            float neighbor2 = 0.0f;

            if (angle >= 157.5f || angle < 22.5f) {
                neighbor1 = gradiant_magnitude[y * width + (x + 1)];
                neighbor2 = gradiant_magnitude[y * width + (x - 1)];
            } else if (angle < 67.5f) {
                neighbor1 = gradiant_magnitude[(y + 1) * width + (x - 1)];
                neighbor2 = gradiant_magnitude[(y - 1) * width + (x + 1)];
            } else if (angle < 112.5f) {
                neighbor1 = gradiant_magnitude[(y + 1) * width + x];
                neighbor2 = gradiant_magnitude[(y - 1) * width + x];
            } else{
                neighbor1 = gradiant_magnitude[(y - 1) * width + (x - 1)];
                neighbor2 = gradiant_magnitude[(y + 1) * width + (x + 1)];
            }
            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                new_buffer[y * width + x] = static_cast<unsigned char>(magnitude);
                non_max_suppressed[y * width + x] = magnitude;
            } else {
                new_buffer[y * width + x] = 0;
                non_max_suppressed[y * width + x] = 0;
            }
        }
    }
    int result = stbi_write_png(filepath.c_str(), width, height, req_comps, new_buffer, width);
    std::cout << "non-max suppression " << (result ? "success" : "fail") << std::endl;
    return non_max_suppressed;
}

std::vector<float> double_thresholding(std::string filepath, std::vector<float> non_max_suppressed, unsigned char lowThreshold, unsigned char highThreshold, int width, int height, int comps){
    int req_comps = 1;
    unsigned char* new_buffer = new unsigned char[width * height]{0};
    std::vector<float> double_thresholded(width * height);
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            unsigned char pixelValue = non_max_suppressed[y * width + x];
            if(pixelValue >= highThreshold){
                new_buffer[y * width + x] = 255; // strong edge
                double_thresholded[y * width + x] = 255.0f;
            } else if(pixelValue >= lowThreshold){
                new_buffer[y * width + x] = 128; // weak edge
                double_thresholded[y * width + x] = 128.0f;
            } else{
                new_buffer[y * width + x] = 0; // non-edge
                double_thresholded[y * width + x] = 0.0f;
            }
        }
    }
    int result = stbi_write_png(filepath.c_str(), width, height, req_comps, new_buffer, width);
    std::cout << "double thresholding " << (result ? "success" : "fail") << std::endl;
    return double_thresholded;
}

void hysteresis(std::string filepath, std::vector<float> double_thresholded, int width, int height, int comps){
    int req_comps = 1;
    unsigned char* new_buffer = new unsigned char[width * height]{0};
    for(int y = 1; y < height - 1; y++){
        for(int x = 1; x < width - 1; x++){
            unsigned char pixelValue = double_thresholded[y * width + x];
            if(pixelValue == 128){ // weak edge
                bool connectedToStrongEdge = false;
                for(int ky = -1; ky <= 1; ky++){
                    for(int kx = -1; kx <= 1; kx++){
                        if(double_thresholded[(y + ky) * width + (x + kx)] == 255){
                            connectedToStrongEdge = true;
                            break;
                        }
                    }
                    if(connectedToStrongEdge) break;
                }
                new_buffer[y * width + x] = connectedToStrongEdge ? 255 : 0;
            } else if(pixelValue == 255){
                new_buffer[y * width + x] = 255; // strong edge
            } else{
                new_buffer[y * width + x] = 0; // non-edge
            }
        }
    }
    int result = stbi_write_png(filepath.c_str(), width, height, req_comps, new_buffer, width);
    std::cout << "canny " << (result ? "success" : "fail") << std::endl;
}

void canny_edge_detection(){
    int width = 256, height = 256, comps = 1;
    std::vector<float> noise_reduced = noise_reduction3x3("res/textures/greyscale.png", "res/textures/noise_reduced.png");
    std::vector<float>* gradiant_angle = new std::vector<float>(width * height);
    std::vector<float> gradiant_magnitude = gradiant_intensity("res/textures/gradient_intensity.png", noise_reduced, gradiant_angle, width, height, comps);
    std::vector<float> non_max_suppressed = non_max_suppression("res/textures/non_max_suppressed.png", gradiant_magnitude, gradiant_angle, width, height, comps);
    std::vector<float> double_thresholded = double_thresholding("res/textures/double_thresholded.png", non_max_suppressed, 50, 100, width, height, comps);
    hysteresis("res/textures/Canny.png", double_thresholded, width, height, comps);
}

void halftone(std::string oldfilepath, std::string newfilepath){
    int width, height, comps;
    int req_comps = 1;
    unsigned char* buffer = stbi_load(oldfilepath.c_str(), &width, &height, &comps, req_comps);
    unsigned char* new_buffer = new unsigned char[width * height * 4]{0};
    int new_width = width * 2;
    int new_height = height * 2;
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            float pixelValue = buffer[y * width + x] / 255.0f;
            if(pixelValue >= 0.2f){
                new_buffer[(y*new_width + x)*2 + new_width]= 255;// bottom left
            }
            if(pixelValue >= 0.4f){
                new_buffer[(y*new_width + x)*2 + 1]= 255;// top right
            }
            if(pixelValue >= 0.6f){
                new_buffer[(y*new_width + x)*2 + new_width + 1]= 255;// bottom right
            }
            if(pixelValue >= 0.8f){
                new_buffer[(y*new_width + x)*2]= 255;// top left
            }
        }    
    }
    
    int result = stbi_write_png(newfilepath.c_str(), new_width, new_height, req_comps, new_buffer, new_width);
    std::cout << "halftone " << (result ? "success" : "fail") << std::endl;
}

int main(void)
{
    grayscale("res/textures/Lenna.png");
    canny_edge_detection();
    halftone("res/textures/greyscale.png", "res/textures/halftone.png");
    return 0;
}