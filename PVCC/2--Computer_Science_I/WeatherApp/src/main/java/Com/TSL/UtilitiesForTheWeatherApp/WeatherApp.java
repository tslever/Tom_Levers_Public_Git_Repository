package Com.TSL.UtilitiesForTheWeatherApp;


import Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForAnInputManager.AnInputManager;
import java.sql.SQLException;


/** **********************************************************************************************************
 * WeatherApp encapsulates the entry point of this program, which listens for commands from a user relating to
 * manipulating locations and displaying weather information.
 *
 * @author Tom Lever
 * @version 1.0
 * @since 07/28/21
 ********************************************************************************************************** */

public class WeatherApp 
{
    /** -------------------------------------------------------------------------------------------------------
     * main is the entry point of this program, which listens for commands from a user relating to manipulating
     * locations and displaying weather information.
     * 
     * @param args
     * @throws SQLException
     ------------------------------------------------------------------------------------------------------- */
    
    public static void main (String[] args) throws SQLException
    {
        AnInputManager theInputManager = new AnInputManager();
        theInputManager.displaysAnIntroduction();
        theInputManager.listensForCommands();
    }
}
