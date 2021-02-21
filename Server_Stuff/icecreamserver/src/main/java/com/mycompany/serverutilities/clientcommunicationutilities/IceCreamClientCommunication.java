
// Allows main to find class IceCreamClientCommunication.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import com.mycompany.serverutilities.productutilities.Answer;
import com.mycompany.serverutilities.productutilities.AnswerBuilder;
import com.mycompany.serverutilities.productutilities.IceCreamProductRetrieval;
import com.
       mycompany.
       serverutilities.
       productutilities.
       CreateStatementException;
import com.
       mycompany.
       serverutilities.
       productutilities.
       ExecuteQueryException;
import com.
       mycompany.
       serverutilities.
       productutilities.
       GetConnectionException;
import com.
       mycompany.
       serverutilities.
       productutilities.
       IceCreamDatabaseNotFoundException;
import com.mycompany.serverutilities.productutilities.Product;
import com.mycompany.serverutilities.productutilities.SearchCriteria;
import org.json.JSONArray;
import org.json.JSONObject;
import java.sql.SQLException;
import java.util.ArrayList;

/**
 * Defines class IceCreamClientCommunication, an instance of which represents
 * an ice cream client communication subsystem. The ice cream client subsystem:
 * 1) stores a server and an ice cream product retrieval subsystem.
 * 
 * 
 * and receives search parameters
 * from the ice cream client, works with an instance of IceCreamProductRetrieval
 * to determine the products that match those search parameters, and provides
 * those matching products to the ice cream client.
 * @version 0.0
 * @author Tom Lever
 */
public class IceCreamClientCommunication {
    
    private final Server server;
    private final IceCreamProductRetrieval retriever;
    
    /**
     * Defines constructor IceCreamClientCommunication, which sets this.server
     * as serverToUse and this.retriever to retrieverToUse.
     * @param serverToUse
     * @param retrieverToUse
     */
    public IceCreamClientCommunication(
        Server serverToUse, IceCreamProductRetrieval retrieverToUse) {
        this.server = serverToUse;
        this.retriever = retrieverToUse;
    }
    
    /**
     * Defines class ControllerToProcessIntoAnswer, whose process method
     * processes an inputToProcess into an Answer.
     */
    private class ControllerToProcessIntoAnswer extends Controller {        
        
        /**
         * Implements Controller's abstract method process, which processes
         * inputToProcess into an Answer.
         * @param messageToProcess 
         */
        @Override
        public Answer process(Object inputToProcess)
            throws ProcessException {            
            
            if (!(inputToProcess instanceof JSONObject)) {
                throw new ProcessException(
                    "inputToProcess must be a JSONObject.");
            }
            
            JSONObject inputToProcessAsJSONObject = (JSONObject)inputToProcess;
            
            if (!inputToProcessAsJSONObject.has("ingredients")) {                
                return AnswerBuilder.buildAnswerWithInfo(
                    "Zero products available: " +
                    "Body of client message had no ingredients list.");
            }
            
            JSONArray ingredientsListAsJSONArray =
               inputToProcessAsJSONObject.getJSONArray("ingredients");
            
            SearchCriteria searchCriteria;
            try {
                searchCriteria = getSearchCriteria(ingredientsListAsJSONArray);
            }
            catch (GetSearchCriteriaRecognizedAHackException e) {
                return AnswerBuilder.buildAnswerWithInfo(
                    "Zero products available: getSearchCriteria recognized a " +
                    "hack.");
            }
            
            try {
                ArrayList<Product> arrayListOfProducts =
                    retriever.getTheProductsMatching(searchCriteria);
                
                // NetBeans wants to split into declaration and assignment.
                Answer answer;
                answer =
                    AnswerBuilder.buildAnswerWithProducts(arrayListOfProducts);
                
                return answer;
            }
            catch (IceCreamDatabaseNotFoundException e) {
                return AnswerBuilder.buildAnswerWithInfo(
                    "Zero products available: Database could not be found in " +
                    "getTheProductsMatching.");
            }
            catch (GetConnectionException e) {
                return AnswerBuilder.buildAnswerWithInfo(
                    "Zero products available: getTheProductsMatching threw " +
                    "GetConnectionException.");
            }
            catch (CreateStatementException e) {
                return AnswerBuilder.buildAnswerWithInfo(
                    "Zero products available: getTheProductsMatching threw " +
                    "CreateStatementException.");
            }
            catch (ExecuteQueryException e) {
                return AnswerBuilder.buildAnswerWithInfo(
                    "Zero products available: getTheProductsMatching threw " +
                    "ExecuteQueryException.");
            }
            catch (SQLException e) {
                return AnswerBuilder.buildAnswerWithInfo(
                    "Zero products available: getTheProductsMatching threw " +
                    "SQLException.");
            }
        }
    }
    
    /**
     * Defines method getSearchCriteria, which gets search criteria from
     * jsonArrayToUse.
     * @param iceCreamApplicationMessageToUse
     * @return new SearchCriteria(ingredientsListAsStringArray)
     * @throws GetSearchCriteriaRecognizedAHackException
     */
    private SearchCriteria getSearchCriteria(JSONArray jsonArrayToUse)
        throws GetSearchCriteriaRecognizedAHackException {

        if (jsonArrayToUse.length() == 0) {
            return new SearchCriteria(new String[0]);
        }

        String[] ingredientsListAsStringArray =
            new String[jsonArrayToUse.length()];
        for (int i = 0; i < ingredientsListAsStringArray.length; i++) {
            ingredientsListAsStringArray[i] = jsonArrayToUse.get(i).toString();
        }
        
        boolean getSearchCriteriaRecognizedAHack = false;
        if (getSearchCriteriaRecognizedAHack) {
            throw new GetSearchCriteriaRecognizedAHackException(
                "getSearchCriteria recognized a hack.");
        }
        
        return new SearchCriteria(ingredientsListAsStringArray);
    }
    
    /**
     * Defines method setMessageInterfaces, which sets the message
     * interfaces of the server.
     * @throws SetMessageInterfacesException 
     */
    public void setMessageInterfaces() throws SetMessageInterfacesException {
        
        MessageHandler messageHandler = new MessageHandler();
        
        ControllerToProcessIntoAnswer controllerToProcessIntoAnswer =
                new ControllerToProcessIntoAnswer();
        
        messageHandler.addController(
            "search-parameters", controllerToProcessIntoAnswer);
        
        MessageInterface messageInterface =
            new MessageInterface("/search-by-ingredients", messageHandler);
        
        this.server.setMessageInterfaces(
            new MessageInterface[]{messageInterface});
    }
    
    public void startServerListeningForMessages()
        throws StartServerListeningForMessagesException {
        
        this.server.startListeningForMessages();
    }
}