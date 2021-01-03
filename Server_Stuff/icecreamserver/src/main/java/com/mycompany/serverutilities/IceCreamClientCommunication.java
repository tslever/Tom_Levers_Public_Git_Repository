
// Allows executable to find class IceCreamClientCommunication.
package com.mycompany.serverutilities;

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
        System.out.println("IceCreamClientCommunication constructor: Started.");
        
        this.server = serverToUse;
        this.retriever = retrieverToUse;
        System.out.println(
            "IceCreamClientCommunication constructor: Set server as " +
            "serverToUse and set retriever as retrieverToUse.");
    }
    
    /**
     * Defines class ExtendedController, which processes ice cream application
     * messages into products.
     */
    private class ExtendedController extends Controller {
        
        /**
         * Defines method process, which overrides Controller's method process
         * and processes ice cream application messages into products.
         * @param messageToProcess 
         */
        @Override
        public void process(IceCreamApplicationMessage messageToProcess) {
            Products products = retriever.getTheProductsMatching(
                getSearchCriteria(messageToProcess));
            server.send(products);
        }
        
        /**
         * Defines method getSearchCriteria, which gets search criteria from
         * an ice cream application message.
         * @param iceCreamApplicationMessageToUse
         * @return new SearchCriteria()
         */
        private SearchCriteria getSearchCriteria(
            IceCreamApplicationMessage iceCreamApplicationMessageToUse) {
            // Functionality to get search criteria from an ice cream
            // application message.
            return new SearchCriteria();
        }
    }

    /**
     * Defines method startProcessingSearchParameters, which sets the message
     * interfaces of the server and and starts the server listening for
     * messages.
     * @throws Exception 
     */
    public void setMessageInterfacesAndStartServerListening() throws Exception {        
        ExtendedController extendedController = new ExtendedController();
        System.out.println(
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Instantiated extendedController, an instance of " +
            "ExtendedController, which is an extension of Controller. " +
            "extendedController is recognized as an instance of Controller.");
        
        MessageHandler messageHandler = new MessageHandler(extendedController);
        System.out.println(
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Instantiated messageHandler based on extendedController.");
        
        MessageInterface messageInterface =
            new MessageInterface("/test", messageHandler);
        System.out.println(
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Instantiated messageInterface, which will connect messages " +
            "received by the part of the server with endpoint '/test' and " +
            "the message handler.");
        
        this.server.setMessageInterfaces(
            new MessageInterface[]{messageInterface});
        // TODO: Generate SetMessageInterfacesException when an Exception is
        // thrown by setMessageInterfaces.
        System.out.println(
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Set messageInterface in array and set array as attribute of " +
            "this.server.");
        
        this.server.startListeningForMessages();
        //startServerListeningForMessages();
        System.out.println(
            "IceCreamClientCommunication." +
            "setMessageInterfacesAndStartServerListening: " +
            "Started server listening for messages.");
    }
    
    ///**
    // * Defines method startServerListeningForMessages, which starts the server
    // * listening for messages.
    // */
    //private void startServerListeningForMessages() throws Exception {
    //    this.server.startListeningForMessages();
    //    // TODO: Throw a ServerIsUnableToListenForMessagesException
    //    // instead of an Exception.
    //}
}