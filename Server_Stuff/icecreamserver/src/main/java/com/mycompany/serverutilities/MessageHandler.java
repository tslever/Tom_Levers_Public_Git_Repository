
// Allows executable to find class MessageHandler.
package com.mycompany.serverutilities;

// Imports classes.
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

/**
 * Defines class MessageHandler whose instances serve as message handlers
 * between message interfaces and controllers.
 * @version 0.0
 * @author Tom Lever
 */
public class MessageHandler implements HttpHandler {
    
    private final Controller controller;
    
    /**
     * Defines constructor MessageHandler which sets attributes of this message
     * handler with inputs.
     * @param controllerToUse
     */
    public MessageHandler(Controller controllerToUse) {
        System.out.println("MessageHandler constructor: Started.");
        
        this.controller = controllerToUse;
        System.out.println(
            "MessageHandler constructor: Sets controller as controllerToUse.");
    }
    
    /**
     * Defines method handle to pick up an HTTP message from a message
     * interface, extracting the ice cream application message in that HTTP
     * message, and handing that ice cream application method to a controller
     * for processing.
     * @param httpExchange
     */
    @Override
    public void handle(HttpExchange httpExchange) {
        System.out.println("MessageHandler.handle: Started.");
        
        IceCreamApplicationMessage iceCreamApplicationMessage =
            getIceCreamApplicationMessage(httpExchange);
        // Must use try block because handle does not throws Exception.
        try {
            this.controller.process(iceCreamApplicationMessage);
            System.out.println(
                "MessageHandler.handle: Got ice cream application message by " +
                "stripping the HTTP layer from the message associated with " +
                "httpExchange. Called this.controller.process.");
        }
        catch(Exception e) {
            System.out.println(
                "BasicController must be extended; process must be " +
                "overridden.");
            System.exit(1);
        }
        // TODO: Catch BasicControllerMustBeExtendedException instead of
        // Exception.
    }
    
    /**
     * Defines getIceCreamApplicationMessage, which gets an ice cream
     * application method from an HTTP message associated with an HttpExchange.
     * @param httpExchangeToUse
     * @return 
     */
    private IceCreamApplicationMessage getIceCreamApplicationMessage(
        HttpExchange httpExchangeToUse) {
        // Functionality to get an ice cream application message from an HTTP
        // message associated with an HttpExchange.
        return new IceCreamApplicationMessage();
    }
}