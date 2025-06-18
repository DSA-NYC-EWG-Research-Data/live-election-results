import { Select as ThemeSelect } from 'theme-ui'
import { Box } from 'theme-ui'

const Select = ({ sx, ...props }) => {
  return (
    <ThemeSelect
      {...props}
      arrow=<Box
        as='svg'
        xmlns='http://www.w3.org/2000/svg'
        width='24'
        height='24'
        viewBox='0 0 24 24'
        fill='currentcolor'
        sx={{
          ml: '-20px',
          mt: '4px',
          alignSelf: 'center',
          pointerEvents: 'none',
        }}
      >
        <path d='M7.41 7.84l4.59 4.58 4.59-4.58 1.41 1.41-6 6-6-6z' />
      </Box>
      sx={{
        fontFamily: 'body',
        fontSize: [3, 3, 3, 3],
        m: [0],
        pl: '1px',
        width: '100%',
        bg: 'transparent',
        '&:focus': {
          outline: 'none',
          borderColor: 'text',
        },
        ...sx,
      }}
    />
  )
}

export default Select
