
// Allows executable to find class IceCreamClientCommunication.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import com.mycompany.serverutilities.productutilities.SearchCriteria;
import com.mycompany.serverutilities.productutilities.IceCreamProductRetrieval;
import com.mycompany.serverutilities.productutilities.Products;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

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
        public void process(IceCreamApplicationMessage messageToProcess) {
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.process: Started."));
            
            SearchCriteria searchCriteria = getSearchCriteria(messageToProcess);
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.process: Got search criteria from " +
                "message to process."));
            
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
        
        /**
         * Defines method getSearchCriteria, which gets search criteria from
         * an ice cream application message.
         * @param iceCreamApplicationMessageToUse
         * @return new SearchCriteria()
         */
        private SearchCriteria getSearchCriteria(
            IceCreamApplicationMessage iceCreamApplicationMessageToUse) {
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.getSearchCriteria: Started."));
            
            // Functionality to get search criteria from an ice cream
            // application message.
            
            logger.log(new LogRecord(Level.INFO,
                "ExtendedController.getSearchCriteria: Returning search " +
                "criteria based on iceCreamApplicationMessageToUse."));
            return new SearchCriteria();
        }
    }

    /**
     * Defines method startProcessingSearchParameters, which sets the message
     * interfaces of the server and and starts the server listening for
     * messages.
     * @throws Exception 
     */
    public void setMessageInterfaces() throws Exception {        
        ExtendedController extendedController = new ExtendedController();
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: Started."));
        
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Instantiated extendedController, an instance of " +
            "ExtendedController, which is an extension of Controller. " +
            "extendedController is recognized as an instance of Controller."));
        
        MessageHandler messageHandler = new MessageHandler(extendedController);
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Instantiated messageHandler based on extendedController."));
        
        MessageInterface messageInterface =
            new MessageInterface("/test", messageHandler);
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Instantiated messageInterface, which will connect messages " +
            "received by the part of the server with endpoint '/test' and " +
            "the message handler."));
        
        this.server.setMessageInterfaces(
            new MessageInterface[]{messageInterface});
        // TODO: Generate SetMessageInterfacesException when an Exception is
        // thrown by setMessageInterfaces.
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Set messageInterface in array and set array as attribute of " +
            "this.server."));
    }
    
    public void startServerListeningForMessages() throws Exception {
        this.server.startListeningForMessages();
        // TODO: Generate ServerStartListeningForMessagesException when an
        // Exception is thrown by startServerListeningForMessages.
        logger.log(new LogRecord(Level.INFO,
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Started server listening for messages."));
    }
}