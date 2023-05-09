--Comments that should be moved to engineering notes:

--Between the Ada specification package_b.ads and Ada body package_b.adb, we are
--illustrating stub-driven development where functions and procedures are
--specified and stubbed.

pragma Style_Checks (Off);
pragma Ada_2012;
with Package_A;

package Package_B is

   procedure Output;
   procedure Initialize_Using(The_Thing_To_Use: Package_A.A_General_Thing);

end Package_B;
