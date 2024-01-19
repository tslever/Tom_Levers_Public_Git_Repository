pragma Ada_2012;


with Ada.Text_IO;


package body Tom_Test is

   ---------------------------
   -- Print_The_Second_Line --
   ---------------------------

   procedure Print_The_Second_Line is

      use Ada.Text_IO;

   begin

      Put_Line("The second line string.");

   end Print_The_Second_Line;

end Tom_Test;
