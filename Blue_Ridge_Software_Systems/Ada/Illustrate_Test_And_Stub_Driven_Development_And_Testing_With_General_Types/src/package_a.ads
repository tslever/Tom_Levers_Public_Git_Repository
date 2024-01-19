--Comments that should be moved to engineering notes:

--Between the Ada specification package_a.ads and Ada body package_a.adb, we are
--illustrating stub-driven development where functions and procedures are
--specified and stubbed.

pragma Style_Checks (Off);
pragma Ada_2012;
with Ada.Strings.Unbounded;

package Package_A is

   type A_General_Thing is private;
   function A_New_General_Thing(The_Base_Value: string) return A_General_Thing; --Functions return values.
   procedure Output(The_Item_To_Output: string; Along_With: A_General_Thing); --Procedures do not return values.
   
private
   type A_General_Thing is record
      Base_Value: Ada.Strings.Unbounded.Unbounded_String;
      end record;

end Package_A;
