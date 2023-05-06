
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
        
        int lengthOfIngredientsList =
            searchParametersToUse.getLengthOfIngredientsList();
        if (lengthOfIngredientsList < 1) {
            return new ArrayList<Product>();
        }
        
        String[] ingredientsList =
            searchParametersToUse.getIngredientsList();
        
        ArrayList<Product> arrayListOfProducts = new ArrayList<>();
        
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

        String sqlQuery =
            "SELECT *\n" +
            "FROM\n" +
            "(SELECT name, image_closed, image_open, description, story, " +
            "productId\n" +
            "FROM\n" +
            "(SELECT index_of_product\n" +
            "FROM\n" +
            "(SELECT id_of_ingredient\n" +
            "FROM Ingredients\n" +
            "WHERE value_of_ingredient in (";

        for (int i = 0; i < lengthOfIngredientsList-1; i++) {
            sqlQuery += "'" + ingredientsList[i] + "', ";
        }
        sqlQuery +=
            "'" +
            ingredientsList[lengthOfIngredientsList-1] +
            "')) AS ids_of_ingredients\n";

        sqlQuery +=
            "JOIN AssociationsBetweenProductsAndIngredients\n" +
            "ON id_of_ingredient = index_of_ingredient\n" +
            "GROUP BY index_of_product\n" +
            "HAVING COUNT(index_of_product) = " +
            lengthOfIngredientsList +
            ") AS indices_of_products_matching_search_parameters\n" +
            "JOIN Products\n" +
            "ON id_of_product = index_of_product) AS " +
            "products_matching_search_parameters_without_arrays";
        
        ResultSet resultSet;
        try {
            resultSet = statement.executeQuery(sqlQuery);
        }
        catch (SQLException e) {
            throw new ExecuteQueryException("executeQuery threw SQLException.");
        }

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
