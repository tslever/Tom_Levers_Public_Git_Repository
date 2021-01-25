
// Allows executable to find class IceCreamApplicationMessage.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

/**
 * Defines class IceCreamApplicationMessage whose instances represent ice cream
 * application messages.
 * @version 0.0
 * @author Tom Lever
 */
class IceCreamApplicationMessage {
    
    private final static Logger logger =
        Logger.getLogger(IceCreamApplicationMessage.class.getName());
    
    private final String bodyOfClientMessage;
    
    /**
     * Defines constructor IceCreamApplicationMethod.
     * @param bodyOfClientMessageToUse
     */
    public IceCreamApplicationMessage(String bodyOfClientMessageToUse) {
        logger.log(new LogRecord(Level.INFO,
            "IceCreamApplicationMessage constructor: Started."));
        
        this.bodyOfClientMessage = bodyOfClientMessageToUse;
        logger.log(new LogRecord(Level.INFO,
            "IceCreamApplicationMessage constructor: Set " +
            "bodyOfClientMessage as bodyOfClientMessageToUse."));
    }
    
    public String getBodyOfClientMessage() {
        return this.bodyOfClientMessage;
    }
}
