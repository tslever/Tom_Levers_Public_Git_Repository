
// Allows main to find class IceCreamProductRetrieval.
package com.mycompany.serverutilities.productutilities;

// Imports classes;
import java.io.File;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;

/**
 * Defines class IceCreamProductRetrieval, an instance of which represents
 * an ice cream product retrieval subsystem, which works with an instance of
 * IceCreamClientCommunication to determine the ice cream products that match
 * specific search parameters.
 * @version 0.0
 * @author Tom Lever
 */
public class IceCreamProductRetrieval {
    
    /**
     * Defines constructor IceCreamProductRetrieval.
     */
    public IceCreamProductRetrieval() { }
    
    /**
     * Defines method getTheProductsMatching to get the ice cream products
     * matching specific search parameters.
     * @param searchParametersToUse
     * @return new Products()
     * @throws IceCreamDatabaseNotFoundException
     * @throws GetConnectionException
     * @throws CreateStatementException
     * @throws ExecuteQueryException
     * @throws SQLException
     */
    public ArrayList<Product> getTheProductsMatching(
        SearchParameters searchParametersToUse)
        throws IceCreamDatabaseNotFoundException,
               GetConnectionException,
               CreateStatementException,
               ExecuteQueryException,
               SQLException {
        
        String pathToIceCreamDatabase = "IceCreamDatabase.sqlite";
        File file = new File(pathToIceCreamDatabase);
        if (!file.exists()) {
            throw new IceCreamDatabaseNotFoundException(
                "Database not found at path.");
        }
        
        String url = "jdbc:sqlite:" + pathToIceCreamDatabase;
        Connection connection;
        try {
            connection = DriverManager.getConnection(url);
        }
        catch (SQLException e) {
            throw new GetConnectionException(
                "getConnection threw SQLException.");
        }
        
        Statement statement;
        try {
            statement = connection.createStatement();
        }
        catch (SQLException e) {
            throw new CreateStatementException(
                "createStatement threw SQLException.");
        }
        
        String sqlQuery = "SELECT * FROM Products";
        ResultSet resultSet;
        try {
            resultSet = statement.executeQuery(sqlQuery);
        }
        catch (SQLException e) {
            throw new ExecuteQueryException("executeQuery threw SQLException.");
        }
        
        ArrayList<Product> arrayListOfProducts = new ArrayList<>();
        while (resultSet.next()) {
            Product product = new Product();
            product.setName(resultSet.getObject("name").toString());
            product.setPathToImageOfClosedProduct(
                resultSet.getObject("image_closed").toString());
            product.setPathToImageOfOpenProduct(
                resultSet.getObject("image_closed").toString());
            product.setDescription(
                resultSet.getObject("description").toString());
            product.setStory(
                resultSet.getObject("story").toString());
            product.setProductId(
                resultSet.getObject("productId").toString());
            arrayListOfProducts.add(product);
        }
        
        return arrayListOfProducts;
    }
}
