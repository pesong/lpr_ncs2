#include "Pipeline.h"


namespace pr {



    const int HorizontalPadding = 4;
    PipelinePR::PipelinePR(std::string detector_filename,
                           std::string finemapping_prototxt, std::string finemapping_caffemodel,
                           std::string segmentationfree_proto,std::string segmentationfree_caffemodel) {
        plateDetection = new PlateDetection(detector_filename);
        fineMapping = new FineMapping(finemapping_prototxt, finemapping_caffemodel);

        segmentationFreeRecognizer =  new SegmentationFreeRecognizer(segmentationfree_proto,segmentationfree_caffemodel);

    }

    PipelinePR::~PipelinePR() {

        delete plateDetection;
        delete fineMapping;

        delete segmentationFreeRecognizer;


    }

    std::vector<PlateInfo> PipelinePR:: RunPiplineAsImage(cv::Mat plateImage) {
        std::vector<PlateInfo> results;
        std::vector<pr::PlateInfo> plates;
        plateDetection->plateDetectionRough(plateImage,plates,36,700);

        for (pr::PlateInfo plateinfo:plates) {

            cv::Mat image_finemapping = plateinfo.getPlateImage();
            image_finemapping = fineMapping->FineMappingVertical(image_finemapping);
            image_finemapping = pr::fastdeskew(image_finemapping, 5);


            //Segmentation-free
            image_finemapping = fineMapping->FineMappingHorizon(image_finemapping, 4, HorizontalPadding+3);
            cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
            plateinfo.setPlateImage(image_finemapping);
            std::pair<std::string,float> res = segmentationFreeRecognizer->SegmentationFreeForSinglePlate(plateinfo.getPlateImage(),pr::CH_PLATE_CODE);
            plateinfo.confidence = res.second;
            plateinfo.setPlateName(res.first);

            results.push_back(plateinfo);
        }

        return results;

    }//namespace pr



}
