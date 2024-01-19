pragma Ada_2012;
procedure Print_Chips_Line is
begin
   pragma Compile_Time_Warning
     (Standard.True, "Print_Chips_Line unimplemented");
   raise Program_Error with "Unimplemented procedure Print_Chips_Line";
end Print_Chips_Line;
