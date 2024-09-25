package AgeGenderCV;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.LINE_AA;
import static org.opencv.imgproc.Imgproc.putText;

public class AgeGenderCV {
    private static final String[] GENDER_LIST = {"Male", "Female"};
    private static final String[] AGE_LIST = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"};
    private static final double[] MODEL_MEAN_VALUES = {78.4263377603, 87.7689143744, 114.895847746};
    private static final double CONFIDENCE_THRESHOLD = 0.4;
    private static int pt = 0;
    
    public static void main(String args[]) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.loadLibrary("opencv_java480");

        // Load networks        
        //System.out.println("Load Networoks");
        Net faceNet = Dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt");
        Net ageNet = Dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel");
        Net genderNet = Dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel");

        // Read frame
        //System.out.println("Read Frame");

        Mat frameFace = Imgcodecs.imread("mag.jpg"); // your data here !
            
        List<Rect> bboxes = new ArrayList<>();
        Size size = new Size(300, 300);
        Scalar scalar = new Scalar(104, 117, 123);

        Mat blob = Dnn.blobFromImage(frameFace, 1.0f, size, scalar, false, false);
        faceNet.setInput(blob);
        Mat detections = faceNet.forward("detection_out");                
        Mat faces = detections.reshape(1, detections.size(2));
        //System.out.println("faces" + detections);

        float[] data = new float[7];
        //System.out.println("face rows: " + faces.rows());
        for (int i = 0; i < faces.size().height; i++) {  
            faces.get(i, 0, data);
            float confidence = data[2];               
            if (confidence > CONFIDENCE_THRESHOLD) {
                int x1 = (int) (faces.get(i, 3)[0] * frameFace.cols());
                int y1 = (int) (faces.get(i, 4)[0] * frameFace.rows());
                int x2 = (int) (faces.get(i, 5)[0] * frameFace.cols());
                int y2 = (int) (faces.get(i, 6)[0] * frameFace.rows());
                bboxes.add(new Rect(x1, y1, x2 - x1, y2 - y1));
                Imgproc.rectangle(frameFace, new org.opencv.core.Point(x1, y1), new org.opencv.core.Point(x2, y2), new org.opencv.core.Scalar(0, 255, 0), (int) Math.round(frameFace.rows() / 150), 8, 0);
            }
        }
        if (!bboxes.isEmpty()) {
            System.out.println("There are "+ bboxes.size() + " faces in the image");
            for (Rect bbox : bboxes) {
                //System.out.println("Processing face");
                Mat face = new Mat(frameFace, bbox);                    
                Size sizeb = new Size(227, 227);
                Scalar scalarb = new Scalar(MODEL_MEAN_VALUES);
                blob = Dnn.blobFromImage(face, 1.0, sizeb, scalarb, false);
                
                //GENDER
                genderNet.setInput(blob);
                
                Mat genderPreds = genderNet.forward();
                int genderIdx = (int) Core.minMaxLoc(genderPreds).maxLoc.x;
                String gender = GENDER_LIST[genderIdx];
                
                //System.out.println("Gender : " + gender + ", conf = " + genderPreds.get(0, 0)[0]);
                //GENDER
                
                
                //AGE
                ageNet.setInput(blob);

                Mat agePreds = ageNet.forward();
                agePreds = agePreds.reshape(1, 1); // reshape to 2D matrix with one row
                int ageIdx = (int) Core.minMaxLoc(agePreds).maxLoc.x;
                String age = AGE_LIST[ageIdx];
                
                //System.out.println("Age : " + age + ", conf = " + agePreds.get(0, ageIdx)[0]);
                if (ageIdx <= 2){
                    pt++;
                    //System.out.println("Pre teen");
                }
                //AGE
                
                //DEBUG OUTPUT
                String label = gender + ", " +age;
                //String label = gender;
                //putText(frameFace, label, new org.opencv.core.Point(bbox.x, bbox.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 255), 2, LINE_AA);
                //Imgcodecs.imwrite("output.jpg", frameFace);
                //DEBUG OUTPUT
            }
            System.out.println("There are "+pt+" pre teen aged detections!");
        }
    }
}