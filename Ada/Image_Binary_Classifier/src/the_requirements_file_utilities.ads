-- TODO: Create a body for this package. In this body, define Requirements_File_Guts.
-- It should be a record (the default for *_Guts).


pragma Ada_2012;


package The_Requirements_File_Utilities is

   type A_Requirements_File is private;
   -- Other things can see and do operations with this type, but cannot see or use implementation details.
   
private
   
   type Requirements_File_Guts;
   type A_Requirements_File is access all Requirements_File_Guts;
   -- access maps to pointer, reference, address

end The_Requirements_File_Utilities;
