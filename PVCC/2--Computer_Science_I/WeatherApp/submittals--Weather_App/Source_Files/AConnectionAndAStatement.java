package Com.TSL.UtilitiesForTheWeatherApp.UtilitiesForADatabaseManager;


import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;


/** *************************************************************************************************************
 * AConnectionAndAStatement represents the structure of an object that encapsulates a connection and a statement.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 07/28/21
 ************************************************************************************************************ */

public class AConnectionAndAStatement
{

    private Connection connection;
    private Statement statement;
    
    
    /** ----------------------------------------------------------------------------------------------------------
     * AConnectionAndAStatement(Connection theConnectionToUse, Statement theStatementToUse) is the two-parameter
     * constructor for AConnectionAndAStatement, which sets this object's connection to a provided connection, and
     * sets this object's statement to a provided statement. 
     * 
     * @param theConnectionToUse
     * @param theStatementToUse
     ---------------------------------------------------------------------------------------------------------- */
    
    public AConnectionAndAStatement(Connection theConnectionToUse, Statement theStatementToUse)
    {
        this.connection = theConnectionToUse;
        this.statement = theStatementToUse;
    }
    
    
    /** -------------------------------------------------------
     * providesItsConnection provides this object's connection.
     * 
     * @return
     ------------------------------------------------------- */
    
    public Connection providesItsConnection()
    {
        return this.connection;
    }
    
    
    /** -----------------------------------------------------
     * providesItsStatement provides this object's statement.
     * 
     * @return
     ----------------------------------------------------- */
    
    public Statement providesItsStatement()
    {
        return this.statement;
    }
    
    
    /** ------------------------------------------------------------
     * close closes the connection and the statement of this object.
     * 
     * @throws SQLException
     ------------------------------------------------------------ */
    
    public void close() throws SQLException
    {
        this.connection.close();
        this.statement.close();
    }
    
}
