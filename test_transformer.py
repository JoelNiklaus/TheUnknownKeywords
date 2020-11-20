def test_transform_(cases, transform_fct):
    errors = 0
    for case in cases:
        transformed = transform_fct(case[0])
        if transformed != case[1]:
            print('Got:  {}\nExp:  {}\n'.format(transformed, case[1]))
            errors += 1
    if errors > 0:
        print('Got {} errors'.format(errors))
    else:
        print('Transformer works')
