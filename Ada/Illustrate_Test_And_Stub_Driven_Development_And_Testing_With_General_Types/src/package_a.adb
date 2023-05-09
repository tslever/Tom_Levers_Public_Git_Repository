--Comments that should be moved to engineering notes:

--Between the Ada specification package_a.ads and Ada body package_a.adb, we are
--illustrating stub-driven development where functions and procedures are
--specified and stubbed.

pragma Style_Checks (Off);
pragma Ada_2012;
with Ada.Strings.Unbounded;
with Ada.Text_IO;

package body Package_A is

   function A_New_General_Thing(The_Base_Value : string) return A_General_Thing
   is

      use Ada.Strings.Unbounded;
      use Ada.Text_IO;
      The_Local_General_Thing: constant A_General_Thing := (Base_Value => To_Unbounded_String(The_Base_Value));

   begin

      Put_Line("In function Package_A.A_New_General_Thing");
      return The_Local_General_Thing;

   end A_New_General_Thing;

   procedure Output(The_Item_To_Output : string; Along_With : A_General_Thing)
   is

      use Ada.Strings.Unbounded;
      use Ada.Text_IO;
      The_General_Thing_To_Use: A_General_Thing renames Along_With;

   begin

      Put_Line("In procedure Package_A.Output");
      Put_Line(The_Item_To_Output & To_String(The_General_Thing_To_Use.Base_Value));
      pragma Compile_Time_Warning(Standard.True, "Output unimplemented");

   end Output;

end Package_A;
