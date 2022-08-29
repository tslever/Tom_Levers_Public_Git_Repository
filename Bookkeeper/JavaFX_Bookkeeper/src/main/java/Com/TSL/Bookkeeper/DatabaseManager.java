package Com.TSL.Bookkeeper;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Stack;
import java.util.regex.Pattern;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;

/** ***************************************************************************************************************
 * DatabaseManager represents the structure for a database manager that allows manipulating a database of accounts.
 * 
 * @author Tom Lever
 * @version 1.0
 * @since 04/21/22
 ************************************************************************************************************** */

public class DatabaseManager {
	
    private final String PATH_TO_FOLDER_FOR_DATABASE = "src/main/resources";
    private final String PATH_TO_DATABASE = this.PATH_TO_FOLDER_FOR_DATABASE + "/Accounts_Database.db";
    
    
    /** ---------------------------------------------------------------------------------------------------------------
     * DatabaseManager() is the zero-parameter constructor for DatabaseManager, which conditionally creates a database.
     * 
     * @throws SQLException
     --------------------------------------------------------------------------------------------------------------- */
    
    public DatabaseManager() throws SQLException {
        this.createDatabase();
    }
    
    
    /** -----------------------------------------------------------------------------------------------------------------------
     * createDatabase creates a folder at a specified path for a database if the folder does not already exist, and creates the
     * database if it doesn't already exist. 
     * 
     * @throws SQLException
     ---------------------------------------------------------------------------------------------------------------------- */
    
    private void createDatabase() throws SQLException {
        File folder = new File(this.PATH_TO_FOLDER_FOR_DATABASE);
        if (!folder.exists()) {
            folder.mkdir();
        }
        File file = new File(this.PATH_TO_DATABASE);
        if (!file.exists()) {
            ConnectionAndStatement connectionAndTheStatement = this.getConnectionAndStatement();
            connectionAndTheStatement.close();
        }
        this.createTablesInDatabase();
    }
    
    
    /** ------------------------------------------------------------------------------------------------------
     * createTablesInDatabase creates a table for account Direct_Deposit_And_CHK, if it doesn't exist already.
     * 
     * @throws SQLException
     ------------------------------------------------------------------------------------------------------ */
    
    private void createTablesInDatabase() throws SQLException
    {
        ConnectionAndStatement connectionAndTheStatement = this.getConnectionAndStatement();
        String query =
            "CREATE TABLE IF NOT EXISTS Direct_Deposit_And_CHK (\n" +
            "ID INTEGER NOT NULL PRIMARY KEY UNIQUE,\n" +
            "Date TEXT NOT NULL,\n" +
            "Name TEXT NOT NULL,\n" +
            "Account_Associated_With_Value TEXT NOT NULL,\n" +
            "Complementary_Account TEXT NOT NULL,\n" +
            "Value NUMERIC NOT NULL\n" +
            ");";
        connectionAndTheStatement.providesItsStatement().execute(query);
        connectionAndTheStatement.close();
    }
    
    
    /** -----------------------------------------------------------------------------
     * display displays the components of the account with the provided account name.
     * 
     * @param account, the name of the account whose components will be displayed
     * @throws SQLException
     ----------------------------------------------------------------------------- */
    
    public void display(String account) throws SQLException {
        ConnectionAndStatement connectionAndStatement = this.getConnectionAndStatement();
        String query = "SELECT ID, Date, Name, Account_Associated_With_Value, Complementary_Account, Value FROM " + account;
        ResultSet resultSet = connectionAndStatement.providesItsStatement().executeQuery (query);
        System.out.println(
            account + "\n" +
            "ID | Date | Name | Account_Associated_With_Value | Complementary_Account | Value"
        );
        while (resultSet.next()) {
            System.out.println(
                resultSet.getInt("ID") + " | " +
                resultSet.getString("Date") + " | " +
                resultSet.getString("Name") + " | " +
                resultSet.getString("Account_Associated_With_Value") + " | " +
                resultSet.getString("Complementary_Account") + " | " +
                resultSet.getFloat("Value")
            );
        }
        connectionAndStatement.close();
    }
    
    
    /** ---------------------------------------------------------------------------------------------------------
     * getLineItems provides an observable list of the line items for the account with the provided account name.
     * 
     * @param account
     * @return lineItems, an observable list of line items
     * @throws ParseException
     * @throws SQLException
     -------------------------------------------------------------------------------------------------------- */
    
    public ObservableList<LineItem> getLineItems(String account) throws ParseException, SQLException {
    	ObservableList<LineItem> lineItems = FXCollections.observableArrayList();
        ConnectionAndStatement connectionAndTheStatement = this.getConnectionAndStatement();
        String query = "SELECT ID, Date, Name, Account_Associated_With_Value, Complementary_Account, Value FROM " + account + " ORDER BY ID DESC";
        ResultSet resultSet = connectionAndTheStatement.providesItsStatement().executeQuery(query);
        while (resultSet.next()) {
            LineItem lineItem = new LineItem(
                resultSet.getInt("ID"),
                new SimpleDateFormat("yyyy-MM-dd").parse(resultSet.getString("Date")),
                resultSet.getString("Name"),
                Account.valueOf(resultSet.getString("Account_Associated_With_Value")),
                Account.valueOf(resultSet.getString("Complementary_Account")),
                resultSet.getFloat("Value")
            );
        	lineItems.add(lineItem);
        }
        connectionAndTheStatement.close();
        return lineItems;
    }
    
    public void migrate(String path, String account) throws IOException, ParseException, SQLException {
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(path))) {
            Stack<LineItem> stack = new Stack<>();
        	String line;
            while ((line = bufferedReader.readLine()) != null) {
                String[] lineItemAsStringArray = line.split(",", 6);
                for (int i = 0; i < lineItemAsStringArray.length; i++) {
                	lineItemAsStringArray[i] = lineItemAsStringArray[i].replaceAll(Pattern.quote("|"), ",");
                }
                stack.push(
                	new LineItem(
                		Integer.parseInt(lineItemAsStringArray[0]),
                		new SimpleDateFormat("yyyy-MM-dd").parse(lineItemAsStringArray[1]),
                		lineItemAsStringArray[2],
                		Account.valueOf(lineItemAsStringArray[3].toUpperCase().replaceAll(" ", "_")),
                		Account.valueOf(lineItemAsStringArray[4].toUpperCase().replaceAll(" ", "_")),
                		Float.parseFloat(lineItemAsStringArray[5])
                	)
                );
            }
            while (!stack.isEmpty()) {
            	LineItem lineItem = stack.pop();
                System.out.println(lineItem);
                insert(account, lineItem);
            }
        }
    }
    
    
    /** ------------------------------------------------------------------------------------
     * inserts inserts into the account with the provided account name a provided line item.
     * 
     * @param account, the account into which the provided line item will be inserted
     * @param lineItem, the line item that will be inserted
     * @throws SQLException
     ----------------------------------------------------------------------------------- */
    
    public void insert(String account, LineItem lineItem) throws SQLException {                
        ConnectionAndStatement connectionAndStatement = this.getConnectionAndStatement();
        String query =
            "INSERT INTO " + account + "\n" +
            "(Date, Name, Account_Associated_With_Value, Complementary_Account, Value)\n" +
            "VALUES ('" +
                new SimpleDateFormat("yyyy-MM-dd").format(lineItem.getDate().getValue()) + "', '" +
                lineItem.getName().getValue() + "', '" +
                lineItem.getAccountAssociatedWithValue().getValue() + "', '" +
                lineItem.getComplementaryAccount().getValue() + "', '" +
                lineItem.getValue().getValue() +
            "');";
        connectionAndStatement.providesItsStatement().execute(query);
        connectionAndStatement.close();
    }
    
    
    /** ------------------------------------------------------------------------------------------
     * getConnectionAndStatement provides a connection and a statement in an encapsulating object.
     * 
     * @return connectionAndStatement
     * @throws SQLException
     ----------------------------------------------------------------------------------------- */
    
    private ConnectionAndStatement getConnectionAndStatement() throws SQLException {
        Connection connection = DriverManager.getConnection("jdbc:sqlite:" + this.PATH_TO_DATABASE);
        Statement statement = connection.createStatement();
        return new ConnectionAndStatement(connection, statement);
    }
}