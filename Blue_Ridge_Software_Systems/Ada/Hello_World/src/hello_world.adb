-- with means this has been compiled into my library of compilation units
-- Library is a set of directories in GNAT project .gpr file (building)
-- Compilation unit is a package or subprogram (i.e., procedure or function)
-- that gets put in library (books).
-- in Java, import means that jar (interface info, bytecode) is in file referenced
-- with means makes visible, but not directly visible

-- use means directly visible
with Ada.Text_IO; -- use Ada.Text_IO;
with Tom_Test;
with Print_Chips_Line;


procedure Hello_World is

   use Ada.Text_IO;
   use Tom_Test;

begin
   Put_Line("Hello World!");
   Print_The_Second_Line;
end Hello_World;
