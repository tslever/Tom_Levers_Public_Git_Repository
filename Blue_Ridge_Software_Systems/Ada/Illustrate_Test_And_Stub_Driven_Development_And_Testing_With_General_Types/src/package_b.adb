--Comments that should be moved to engineering notes:

--Between the Ada specification package_b.ads and Ada body package_b.adb, we are
--illustrating stub-driven development where functions and procedures are
--specified and stubbed.

pragma Style_Checks (Off);
pragma Ada_2012;
with Ada.Text_IO;
with Package_A;

package body Package_B is

   The_Local_Thing : Package_A.A_General_Thing;
   This_Package_Is_Uninitialized : Boolean := True;

   procedure Initialize_Using(The_Thing_To_Use: Package_A.A_General_Thing)
   is

      use Ada.Text_IO;

   begin

      Put_Line("In procedure Package_B.Initialize_Using");
      The_Local_Thing := The_Thing_To_Use;
      This_Package_Is_Uninitialized := False;
      pragma Compile_Time_Warning(Standard.True, "Initialize_Using unimplemented");

   end Initialize_Using;

   procedure Output
   is

      use Ada.Text_IO;

   begin

      Put_Line("In procedure Package_B.Output");
      if This_Package_Is_Uninitialized then
         raise Program_Error;
      end if;
      Package_A.Output("Output of procedure Package_B.Output: A string hard coded in Ada body package_b.adb and ", Along_With => The_Local_Thing);
      pragma Compile_Time_Warning(Standard.True, "Output unimplemented");

   end Output;

end Package_B;
