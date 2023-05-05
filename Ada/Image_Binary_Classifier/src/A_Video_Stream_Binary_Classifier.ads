pragma Ada_2012;


-- A video stream is running constantly.
-- Every The_Sampling_Period, an image frame is extracted from the video stream.
-- The result of image binary classification is time-stamped and stored.

-- When A_Match_Was_Recently_Found is invoked, it returns true when at least one
-- stored result has a value of true within The_Match_Validity_Time duration.

generic
   The_Match_Validity_Time : Duration;
   The_Sampling_Period : Natural;
   The_Video_Stream_To_Use : in A_Video_Stream;
   with package The_Image_Binary_Classifier_To_Use is new An_Image_Binary_Classifier(<>);
   -- the package has to be an instance of An_image_Binary_Classifier, and I'm not putting any limits on what can be used to instantiate it.
   -- Instantiating a package means

package A_Video_Stream_Binary_Classifier is

   function A_Match_Was_Recently_Found return Boolean;

end A_Video_Stream_Binary_Classifier
