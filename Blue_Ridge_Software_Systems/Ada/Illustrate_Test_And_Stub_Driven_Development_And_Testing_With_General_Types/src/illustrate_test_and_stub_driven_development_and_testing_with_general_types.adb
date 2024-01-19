--Comments that should be moved to engineering notes:

--In this Ada body, compiling this body with The_Test_Object.print illustrates
--test-driven development where objects and their functions / procedures are
--called before they are defined. This will provide us a list of packages to
--stub out first and then to implement.

--Compiling this body with the other code in the begin block illustrates,
--in present implementation, defining a concrete object The_Thing_To_Use of type
--A_General_Thing, a notional object; initializing Package_B with
--The_Thing_To_Use; and outputting a fixed string along with the concrete object
--with which the package was initialized.

--with statements make packages directly visible to this compilation unit,
--which is this Ada body.
with Ada.Text_IO;
with Package_A;
with Package_B;

procedure Illustrate_Test_And_Stub_Driven_Development_And_Testing_With_General_Types is

   The_Thing_To_Use: Package_A.A_General_Thing;

begin

   --The_Test_Object.print

   The_Thing_To_Use := Package_A.A_New_General_Thing("the string representation of The_Thing_To_Use in Ada body hello.adb");
   --If Package_B.Initialize_Using(The_Thing_To_Use) is commented,
   --a program error will be raised.
   Package_B.Initialize_Using(The_Thing_To_Use);
   Package_B.Output;

exception
   when others => Ada.Text_IO.Put_Line("An error occurred.");
   --because Package_B was not initialized using The_Thing_To_Use, in the
   --present implementation.

end Illustrate_Test_And_Stub_Driven_Development_And_Testing_With_General_Types;
