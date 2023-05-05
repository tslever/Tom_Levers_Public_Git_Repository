pragma Ada_2012;


-- A_Requirements_File and An_Image must be Types within a package in the Ada library.
with The_Image_Utilities;             use The_Image_Utilities; -- with provides visibility; use provides direct visibility
with The_Requirements_File_Utilities; use The_Requirements_File_Utilities;


generic
   The_Requirements_File_To_Use : in A_Requirements_File; -- This syntax means Object : Type
   -- in means A_Requirements_File can't be changed by anything within package An_Image_Binary_Classifier

package An_Image_Binary_Classifier is

   function A_Match_Exists_For(The_Image_To_Check : in An_Image) return Boolean;
   -- in means The_Match_Exists_For can't be changed within the scope of A_Match_Exists_For

end An_Image_Binary_Classifier;
