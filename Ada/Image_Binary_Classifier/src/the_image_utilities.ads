pragma Ada_2012;


package The_Image_Utilities is

   type An_Image is private;
   -- Other things can see and do operations with this type, but cannot see or use implementation details.
   
private
   
   type An_Image_Guts;
   type An_Image is access all An_Image_Guts;
   -- access maps to pointer, reference, address
   -- all means an object of type An_Image_Guts or any object of an inheriting type of type An_Image_Guts (i.e., a child of type An_Image_Guts)
   -- An_Image is a reference to an object of type An_Image_Guts or a child type of An_Image_Guts.
   
end The_Image_Utilities;
