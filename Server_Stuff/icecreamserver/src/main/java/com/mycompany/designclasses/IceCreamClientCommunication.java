
/*
 * Distinguishes this design class from IceCreamClientCommunication in
 * package com.mycompany.serverutilities.
 */
package com.mycompany.designclasses;

/**
 * Defines class IceCreamClientCommunication, an instance of which represents
 * an ice cream client communication subsystem and receives search parameters
 * from the ice cream client, works with an instance of IceCreamProductRetrieval
 * to determine the products that match those search parameters, and provides
 * those matching products to the ice cream client.
 * @author Tom Lever
 */
public class IceCreamClientCommunication {
    
    // Commented out to allow building.
    //private final Server server;
    //private final IceCreamProductRetrieval retriever;
    
    /**
     * Defines constructor IceCreamClientCommunication which sets attributes of
     * this iceCreamClientCommunication with inputs.
     * inputs.
     * @param serverToUse
     * @param retrieverToUse
     */
    // Commented out to allow building.
    //public IceCreamClientCommunication(
    //    Server serverToUse, IceCreamProductRetrieval retrieverToUse) { }
    
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
        public void process(IceCreamApplicationMessage messageToProcess) { }
        
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
    public void startProcessingSearchParameters() throws Exception { }
    
    /**
     * Defines method startServerListeningForMessages, which starts the server
     * listening for messages.
     */
    private void startServerListeningForMessages() throws Exception { }
}
