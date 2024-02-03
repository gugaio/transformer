class ModelInfo:
    @staticmethod
    def print(model, logger):
        logger.info('\n\n***** Model Info *****')        
        total_param_size = 0
        total_parametres = 0
        for name, param in model.named_parameters():
            param_size = (param.nelement() * param.element_size())
            param_size_mb = param_size / 1024**2
            logger.info('Name: {} Total parameters: {} Size: {:.3f}MB'.format(name, param.nelement() , param_size_mb))            
            total_param_size += param_size
            total_parametres += param.nelement()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (total_param_size + buffer_size) / 1024**2

        logger.info('\n\nSummary:')      
        logger.info('Total parameters: {}'.format((total_parametres)))
        logger.info('model size: {:.3f}MB'.format(size_all_mb))
