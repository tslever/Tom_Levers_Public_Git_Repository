package brss.type_definer_and_assigner;

public class Main 
{
	
    public static void main( String[] args )
    	throws NotDivisibleByOneHundredthException,
    		   TemperatureValueLessThanLowerLimitException,
    		   TemperatureValueGreaterThanUpperLimitException
    {
        A_Temperature my_temperature = new A_Temperature();
        
        //my_temperature = 100.0;
        // Eclipse throws compile-time error "Type mismatch: cannot convert from double to A_Temperature".
        
        my_temperature.Takes_A_Value_Of(100.0); // Valid
        
        //my_temperature.Takes_A_Value_Of(-20000.0);
        // Program throws runtime TemperatureValueLessThanLowerLimitException
        
        my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Scale.CELSIUS, 0.0); // Valid
        
        my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Scale.FAHRENHEIT, 0.0); // Valid
        
        //my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Scale.KELVIN, 0.0);
        // Eclipse throws compile-time error "KELVIN cannot be resolved or is not a field".
        
        //my_temperature.Takes_A_Unit_And_A_Value_Of(A_Temperature_Scale.CELSIUS, -302.00);
        // Program throws runtime TemperatureValueLessThanLowerLimitException.
        
    }
}
