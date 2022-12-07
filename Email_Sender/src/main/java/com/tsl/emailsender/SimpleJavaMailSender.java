package com.tsl.emailsender;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

import org.simplejavamail.api.email.AttachmentResource;
import org.simplejavamail.api.email.Email;
import org.simplejavamail.api.mailer.Mailer;
import org.simplejavamail.api.mailer.config.TransportStrategy;
import org.simplejavamail.email.EmailBuilder;
import org.simplejavamail.mailer.MailerBuilder;
import org.simplejavamail.mailer.internal.MailerRegularBuilderImpl;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import jakarta.activation.DataSource;

/**
 * An {@code SimpleJavaMailSender} object represents an Simple Java Mail sender.
 */
public class SimpleJavaMailSender extends EmailSender {

    /**
     * Constructs a Simple Java Mail sender based on a {@code Configurations} object.
     * 
     * @param configurationsToUse a {@code Configurations} object with which to construct a Simple Java Mail sender
     * @param emailInfoToUse an {@code EmailInfo} object with which to construct a Simple Java Mail Sender
     * @throws IOException if email information cannot be gotten
     */
    public SimpleJavaMailSender(Configurations configurationsToUse, EmailInfo emailInfoToUse) throws IOException {
        super(configurationsToUse, emailInfoToUse);
    }
    
    /**
     * Sends an email using Simple Java Mail.
     * 
     * @throws IOException if an email header or body cannot be read
     */
    public void send() throws IOException {
        ArrayList<AttachmentResource> arrayListOfAttachmentResources = new ArrayList<>();
        for (DataSource dataSource : emailInfo.getListOfAttachments()) {
            AttachmentResource attachmentResource = new AttachmentResource(null, dataSource, null, null);
            arrayListOfAttachmentResources.add(attachmentResource);
        }
        Email email =
            EmailBuilder
                .startingBlank()
                .from(emailInfo.getFromAddress())
                .toMultiple(emailInfo.getListOfToAddresses())
                .withSubject(emailInfo.getSubject())
                .withPlainText(emailInfo.getBody())
                .withAttachments(arrayListOfAttachmentResources)
                .buildEmail();
        Path pathToServerConfiguration = Path.of(configurations.getServerConfigurationDirectory() + "/server-configuration.json");
        String serverConfigurationAsString = Files.readString(pathToServerConfiguration, StandardCharsets.UTF_8);
        JsonObject serverConfiguration = JsonParser.parseString(serverConfigurationAsString).getAsJsonObject();
        String domain = serverConfiguration.get("domain").getAsString();
        int port = serverConfiguration.get("port").getAsInt();
        Path pathToBasicCredentials = Path.of(configurations.getCredentialsDirectory() + "/basic-credentials.json");
        String basicCredentialsAsString = Files.readString(pathToBasicCredentials, StandardCharsets.UTF_8);
        JsonObject basicCredentials = JsonParser.parseString(basicCredentialsAsString).getAsJsonObject();
        String username = basicCredentials.get("username").getAsString();
        String password = basicCredentials.get("password").getAsString();
        MailerRegularBuilderImpl mailerRegularBuilderImpl =
            MailerBuilder
                .withSMTPServer(domain, port, username, password)
                .withSessionTimeout(10 * 1000)
                .withTransportStrategy(TransportStrategy.SMTPS);
        Mailer mailer = mailerRegularBuilderImpl.buildMailer();
        mailer.sendMail(email);
    }
}