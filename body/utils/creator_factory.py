

def makeClassMaker(the_class, *args, **kwargs):
    def makeClass(*other_args, **other_kwargs):
        return the_class(*other_args, *args, **kwargs, **other_kwargs)
    return makeClass 