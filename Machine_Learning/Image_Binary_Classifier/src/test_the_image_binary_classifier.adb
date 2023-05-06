pragma Ada_2012;


with Ada.Text_IO;
with An_Image_Binary_Classifier;
with The_Image_Utilities;
with The_Requirements_File_Utilities;


procedure Test_The_Image_Binary_Classifier is

   use Ada.Text_IO;
   use The_Image_Utilities;
   use The_Requirements_File_Utilities;

   The_Cat_Requirements_File : A_Requirements_File;
   The_Cat_Image_To_Test : An_Image;
   package The_Cat_Classifier is new An_Image_Binary_Classifier(The_Requirements_File_To_Use => The_Cat_Requirements_File);

   The_Oak_Leaf_Requirements_File : A_Requirements_File;
   The_Oak_Leaf_Image_To_Test : An_Image;
   package The_Oak_Leaf_Classifier is new An_Image_Binary_Classifier(The_Requirements_File_To_Use => The_Oak_Leaf_Requirements_File);
   function An_Oak_Leaf_Match_Exists_For(The_Image_To_Use : in An_Image) return Boolean renames The_Oak_Leaf_Classifier.A_Match_Exists_For;

begin

   if The_Cat_Classifier.A_Match_Exists_For(The_Cat_Image_To_Test) then
      Put_Line("A match exists.");
   end if;

   if An_Oak_Leaf_Match_Exists_For(The_Oak_Leaf_Image_To_Test) then
      Put_Line("A match exists.");
   end if;

   null;
end Test_The_Image_Binary_Classifier;
