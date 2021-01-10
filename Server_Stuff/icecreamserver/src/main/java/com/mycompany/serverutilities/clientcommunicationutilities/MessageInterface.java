
// Allows executable to find class MessageInterface.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Defines class MessageInterface that encapsulates functionality to store and
 * get an endpoint and a messageHandler.
 * @version 0.0
 * @author Tom Lever
 */
class MessageInterface {
    
    private final static Logger logger =
        Logger.getLogger(MessageInterface.class.getName());
    
    private final String endpoint;
    private final MessageHandler messageHandler;
    
    /**
     * Defines constructor MessageInterface which sets attributes of this
     * message interface with inputs.
     * @param endpointToUse
     * @param messageHandlerToUse
     */
    public MessageInterface(
        String endpointToUse, MessageHandler messageHandlerToUse) {
        logger.log(new LogRecord(Level.INFO,
            "MessageInterface constructor: Started."));
        
        this.endpoint = endpointToUse;
        this.messageHandler = messageHandlerToUse;
        logger.log(new LogRecord(Level.INFO,
            "MessageInterface constructor: Set endpoint as endpointToUse " +
            "and set messageHandler as messageHandlerToUse."));
    }
    
    /**
     * Defines method getEndpoint which allows a calling method to get a
     * endpoint stored as an attribute of this MessageInterface object.
     * @return this.endpoint
     */
    public String getEndpoint() {
        logger.log(new LogRecord(Level.INFO,
            "MessageInterface.getEndpoint: Started."));
        
        logger.log(new LogRecord(Level.INFO,
            "MessageInterface.getEndpoint: Returning this.endpoint."));
        return this.endpoint;
    }

    /**
     * Defines method getMessageHandler which allows a calling method to get a
     * messageHandler stored as an attribute of this MessageInterface object.
     * @return this.messageHandler
     */
    public MessageHandler getMessageHandler() {
        logger.log(new LogRecord(Level.INFO,
            "MessageInterface.getMessageHandler: Started."));
        
        logger.log(new LogRecord(Level.INFO,
            "MessageInterface.getMessageHandler: " +
            "Returning this.messageHandler."));
        return this.messageHandler;
    }
}
