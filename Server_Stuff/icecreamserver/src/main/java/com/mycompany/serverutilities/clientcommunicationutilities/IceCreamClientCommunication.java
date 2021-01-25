
// Allows executable to find class IceCreamClientCommunication.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import com.mycompany.serverutilities.productutilities.SearchCriteria;
import com.mycompany.serverutilities.productutilities.IceCreamProductRetrieval;
import com.mycompany.serverutilities.productutilities.Products;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Defines class IceCreamClientCommunication, an instance of which represents
 * an ice cream client communication subsystem and receives search parameters
 * from the ice cream client, works with an instance of IceCreamProductRetrieval
 * to determine the products that match those search parameters, and provides
 * those matching products to the ice cream client.
 * @version 0.0
 * @author Tom Lever
 */
public class IceCreamClientCommunication {
    
    private final static Logger logger =
        Logger.getLogger(IceCreamClientCommunication.class.getName());
    
    private final Server server;
    private final IceCreamProductRetrieval retriever;
    
    /**
     * Defines constructor IceCreamClientCommunication which sets attributes of
     * this iceCreamClientCommunication with inputs.
     * inputs.
     * @param serverToUse
     * @param retrieverToUse
     */
    public IceCreamClientCommunication(
        Server serverToUse, IceCreamProductRetrieval retrieverToUse) {
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication constructor: Started."));
        
        this.server = serverToUse;
        this.retriever = retrieverToUse;
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication constructor: Set server as " +
            "serverToUse and set retriever as retrieverToUse."));
    }
    
    /**
     * Defines class ExtendedController, which processes ice cream application
     * messages into products.
     */
    private class ExtendedController extends Controller {
        
        /**
         * Implements Controller's abstract method process,
         * which processes ice cream application messages into products.
         * @param messageToProcess 
         */
        @Override
        public void process(Object valuesToUse) {            
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.process: Started."));
            
            JSONObject valuesToUseAsJSONObject = (JSONObject)valuesToUse;
            
            if (valuesToUseAsJSONObject.has("invalid-key") &&
                valuesToUseAsJSONObject.get("invalid-key").toString().equals(
                    "no-valid-keys")) {
                logger.log(new LogRecord(Level.INFO,
                    "ExtendedController.process: Body of client message had " +
                    "no valid keys: sending an empty products list back."));
                server.send(new Products());
                return;
            }
            
            if (!valuesToUseAsJSONObject.has("ingredients")) {
                logger.log(new LogRecord(Level.INFO,
                    "ExtendedController.process: Body of client message had " +
                    "no ingredients list: sending an empty products list " +
                    "back."));
                server.send(new Products());
                return;
            }
            
            JSONArray ingredientsListAsJSONArray =
               valuesToUseAsJSONObject.getJSONArray("ingredients");
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.process: Got ingredientsListAsJSONArray '" +
                ingredientsListAsJSONArray.toString() + "'."));
            
            SearchCriteria searchCriteria =
                getSearchCriteria(ingredientsListAsJSONArray);
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.process: Got search criteria from " +
                "values to use."));
            
            Products products =
                retriever.getTheProductsMatching(searchCriteria);
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.process: Got the products matching " +
                "the search criteria."));
            
            server.send(products);
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.process: Had server send products to " +
                "client."));
        }
    }
    
    /**
     * Defines method getSearchCriteria, which gets search criteria from
     * an ice cream application message.
     * @param iceCreamApplicationMessageToUse
     * @return new SearchCriteria()
     */
    private SearchCriteria getSearchCriteria(
        JSONArray jsonArrayToUse) {
        logger.log(new LogRecord(Level.INFO,
            "ExtendedController.getSearchCriteria: Started."));

        if (jsonArrayToUse.length() == 0) {
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.getSearchCriteria: Returning " +
                "new SearchCriteria, passed a new empty string array, " +
                "because there are no ingredients in the " +
                "ingredientsListAsJSONArray."));
            return new SearchCriteria(new String[0]);
        }

        String[] ingredientsListAsStringArray =
            new String[jsonArrayToUse.length()];
        for (int i = 0;
             i < ingredientsListAsStringArray.length;
             i++) {
            ingredientsListAsStringArray[i] =
                jsonArrayToUse.get(i).toString();
        }
        logger.log(new LogRecord(Level.INFO,
            "ExtendedController.getSearchCriteria: Created " +
            "ingredientsListAsStringArray '" +
            Arrays.toString(
                ingredientsListAsStringArray) + "'."));

        logger.log(new LogRecord(Level.INFO,
            "ExtendedController.getSearchCriteria: Returning new " +
            "SearchCriteria based on ingredientsListAsStringArray."));
        return new SearchCriteria(ingredientsListAsStringArray);
    }

    /**
     * Defines method startProcessingSearchParameters, which sets the message
     * interfaces of the server and and starts the server listening for
     * messages.
     * @throws Exception 
     */
    public void setMessageInterfaces() throws Exception {
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.setMessageInterfaces: Started."));
        
        MessageHandler messageHandler = new MessageHandler();
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.setMessageInterfaces: " +
            "Instantiated messageHandler."));
        
        ExtendedController extendedControllerForProcessingSearchParameters =
            new ExtendedController();        
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.setMessageInterfaces: " +
            "Instantiated extendedControllerForProcessingSearchParameters, " +
            "an instance of ExtendedController, which is an extension of " +
            "Controller. This instance is recognized as an instance of " +
            "Controller."));
        
        messageHandler.addController(
            "search-parameters",
            extendedControllerForProcessingSearchParameters);
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.setMessageInterfaces: " +
            "Added extendedControllerForProcessingSearchParameters to " +
            "messageHandler along with the key 'search-parameters', which " +
            "MessageHandler.handle will use to identify the appropriate " +
            "Controller when it finds 'search-parameters=<values to process>' "
            + "in the body of a client message."));
        
        ExtendedController extendedControllerUsedWhenKeyIsInvalid =
             new ExtendedController();
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.setMessageInterfaces: " +
            "Instantiated extendedControllerUsedWhenKeyIsInvalid."));
        
        messageHandler.addController(
            "invalid-key", extendedControllerUsedWhenKeyIsInvalid);
        
        MessageInterface messageInterface =
            new MessageInterface("/test", messageHandler);
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.setMessageInterfaces: " +
            "Instantiated messageInterface, which will connect messages " +
            "received by the part of the server with endpoint '/test' and " +
            "the appropriate message handler."));
        
        this.server.setMessageInterfaces(
            new MessageInterface[]{messageInterface});
        // TODO: Generate SetMessageInterfacesException when an Exception is
        // thrown by setMessageInterfaces.
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.setMessageInterfaces: " +
            "Set messageInterface in an array and set the array as an " +
            "attribute of this.server."));
    }
    
    public void startServerListeningForMessages() throws Exception {
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.startServerListeningForMessages: " +
            "Started."));
        
        this.server.startListeningForMessages();
        // TODO: Generate ServerStartListeningForMessagesException when an
        // Exception is thrown by startServerListeningForMessages.
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication.startServerListeningForMessages: " +
            "Started server listening for messages."));
    }
}