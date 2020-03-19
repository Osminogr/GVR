#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// --------------------------------------------------------------------------------------



int main(int argc, char** argv)
{
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
    anet_type net;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

    //вносим в массив лица из аргументов
    matrix<rgb_pixel> img;
    load_image(img, argv[1]);
    image_window win1(img);
    // загружаем лица людей по аргументам
    std::vector<matrix<rgb_pixel>> recogitions;
    for (int i = 2; i < argc; ++i)
    {
        matrix<rgb_pixel> img_proc;
        load_image(img_proc, argv[i]);
        for (auto face : detector(img_proc))
        {
            auto shape = sp(img_proc, face);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img_proc, get_face_chip_details(shape, 150, 0.25), face_chip);
            win1.add_overlay(face);
            recogitions.push_back(move(face_chip));
        }
    }

    std::vector<matrix<float, 0, 1>> face_descriptor_d = net(recogitions);




    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;
        image_window win2;

        
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            cv_image<bgr_pixel> cimg(temp);
            std::vector<matrix<bgr_pixel>> faces;
            for (auto face : detector(img))
            {
                auto shape = sp(cimg, face);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));
                win2.set_image(face_chip);
            }

            if (faces.size() != 0)
            {
                cout << "smome" << endl;
            }

            std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);

            for (size_t i = 0; i < face_descriptors.size(); ++i)
            {
                for (size_t j = i; j < face_descriptor_d.size(); ++j) {
                    if (length(face_descriptors[i] - face_descriptor_d[j]) < 0.6) {
                        cout << "person here" << endl;
                        cout << j << endl;
                    }
                }
            }


            win.set_image(cimg);
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
